from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np


from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# --- Episodes ---
NUM_EPISODES    = 2000
SCENARIOS       = ['training_1', 'training_2', 'training_3']
SUMMARY_STEP_SIZE = 50

# Curriculum phases (episode thresholds)
PHASE_EASY      = NUM_EPISODES // 5        # 0   → 400  : short goals, no random start
PHASE_MEDIUM    = (2 * NUM_EPISODES) // 5  # 400 → 800  : random start enabled
PHASE_HARD      = (3 * NUM_EPISODES) // 4  # 800 → 1500 : random velocity enabled
# > 1500 : full randomization + wind start offset

# --- PPO ---
GAMMA           = 0.995
GAE_LAMBDA      = 0.95

LR              = 3e-4

UPDATE_TIMESTEP = 4096
PPO_EPOCHS      = 10
MINIBATCH_SIZE  = 256

CLIP_EPS        = 0.2
VALUE_CLIP      = 0.2

ENTROPY_COEF    = 0.03   # ↑ légèrement pour encourager l'exploration
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5

CROP_SIZE       = 17   # ↑ 9→17 : le bateau voit 8 cases devant lui, suffisant pour anticiper l'île

# --- Reward Shaping ---
COLLISION_PENALTY       = -25.0
STEP_PENALTY            = -0.02
DIST_DELTA_REWARD       = 0.10
HEADING_REWARD_COEF     = 0.05
MILESTONE_BONUS         = 2.0
SUCCESS_BONUS           = 20.0


# ---------------------------------------------------------------------------
# Architecture : SailingNet amélioré
# ---------------------------------------------------------------------------

class SailingNet(nn.Module):
    """
    Architecture CNN pour l'encodage du crop local.

    Pourquoi CNN et non MLP ?
    Un MLP aplati traite les 243 pixels comme une liste non ordonnée : il ne peut
    pas apprendre facilement "obstacle à gauche = ne pas aller à gauche" car
    aucune relation de voisinage n'est encodée structurellement.
    Un CNN 2D capte ces relations spatiales avec ~10x moins de paramètres et
    converge beaucoup plus vite sur des tâches de navigation locale.

    Le CNN ici est intentionnellement minimaliste (2 conv 3×3) pour rester rapide
    sur CPU. Sur un crop 9×9 il produit un vecteur de taille 32 qui résume
    l'environnement local de façon spatialement cohérente.
    """

    def __init__(self, crop_size=9, n_actions=9, n_scalars=9):
        super().__init__()

        # --- Encodeur spatial CNN (entrée : 3 × crop × crop) ---
        # Conv1 : 3→16 filtres 3×3, padding=1 → conserve la taille (9×9)
        # Conv2 : 16→32 filtres 3×3, sans padding → 7×7
        # AdaptiveAvgPool : → 3×3 fixe quelle que soit crop_size
        # Flatten → 32*3*3 = 288 → Linear → 64
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # → (16, 9, 9)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),  # → (32, 7, 7)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                  # → (32, 3, 3)
            nn.Flatten(),                                  # → 288
            nn.Linear(288, 64),
            nn.ReLU(),
        )

        # --- Encodeur scalaire (inchangé fonctionnellement) ---
        self.scalar_encoder = nn.Sequential(
            nn.Linear(n_scalars, 48),
            nn.Tanh(),
        )

        # --- Fusion et têtes actor/critic ---
        self.shared = nn.Sequential(
            nn.Linear(64 + 48, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        self.actor  = nn.Linear(64, n_actions)
        self.critic = nn.Linear(64, 1)

    def forward(self, local_map, scalars):
        x_map    = self.map_encoder(local_map)   # (B, 3, H, W) → (B, 64)
        x_scalar = self.scalar_encoder(scalars)  # (B, 9) → (B, 48)
        x        = torch.cat([x_map, x_scalar], dim=1)
        x        = self.shared(x)
        return self.actor(x), self.critic(x)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory:
    def __init__(self):
        self.states_map      = []
        self.states_scalars  = []
        self.actions         = []
        self.logprobs        = []
        self.rewards         = []
        self.is_terminals    = []
        self.values          = []

    def clear(self):
        self.states_map.clear()
        self.states_scalars.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.values.clear()


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOSailingAgent:
    def __init__(self, crop_size=CROP_SIZE):
        self.crop_size = crop_size
        self.policy     = SailingNet(crop_size=crop_size)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=LR, eps=1e-5)
        self.scheduler  = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_EPISODES
        )
        self.policy_old = SailingNet(crop_size=crop_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.loss_history = {'actor': [], 'critic': []}

    # ------------------------------------------------------------------
    # Extraction du crop local
    # ------------------------------------------------------------------

    def get_local_crop(self, obs):
        pos    = obs[0:2].astype(int)
        wmap   = obs[32774:49158].reshape(128, 128)
        wfield = obs[6:32774].reshape(128, 128, 2)
        pad    = self.crop_size // 2

        wmap_p = np.pad(wmap,   pad, constant_values=1)
        wf_p   = np.pad(wfield, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)

        y, x = pos[1] + pad, pos[0] + pad
        crop  = np.zeros((3, self.crop_size, self.crop_size))
        crop[0]  = wmap_p[y-pad : y+pad+1, x-pad : x+pad+1]
        crop[1:] = wf_p  [y-pad : y+pad+1, x-pad : x+pad+1].transpose(2, 0, 1)

        return torch.FloatTensor(crop).unsqueeze(0)

    # ------------------------------------------------------------------
    # Sélection d'action
    # ------------------------------------------------------------------

    def select_action(self, obs, goal, prev_angle):
        with torch.no_grad():
            m_input = self.get_local_crop(obs)

            curr_w  = obs[4:6]
            curr_a  = np.arctan2(curr_w[1], curr_w[0])

            da = 0.0 if prev_angle is None else np.arctan2(
                np.sin(curr_a - prev_angle),
                np.cos(curr_a - prev_angle)
            )

            vel_x = obs[2] / 10.0
            vel_y = obs[3] / 10.0

            dx = goal[0] - obs[0]
            dy = goal[1] - obs[1]

            # --- nouvelles features directionnelles ---
            dist_to_goal   = np.sqrt(dx**2 + dy**2)
            angle_to_goal  = np.arctan2(dy, dx)          # [-π, π]
            dx_norm        = dx / 128.0
            dy_norm        = dy / 128.0
            dist_norm      = dist_to_goal / 181.0         # diagonale max ≈ 181

            x_norm = obs[0] / 128.0
            y_norm = obs[1] / 128.0

            s_input = torch.FloatTensor([
                vel_x,
                vel_y,
                dx_norm,
                dy_norm,
                da / np.pi,
                x_norm,
                y_norm,
                np.sin(angle_to_goal),   # représentation polaire stable
                dist_norm,
            ]).unsqueeze(0)

            logits, value = self.policy_old(m_input, s_input)
            dist          = Categorical(logits=logits)
            action        = dist.sample()

            return (
                action.item(),
                dist.log_prob(action),
                value.squeeze(),
                curr_a,
                angle_to_goal,   # retourné pour le reward shaping
                m_input,
                s_input,
            )

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(self, rewards, dones, values, next_value):
        advantages = []
        gae        = 0
        values     = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + GAMMA * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    # ------------------------------------------------------------------
    # Mise à jour PPO
    # ------------------------------------------------------------------

    def update(self, memory):
        with torch.no_grad():
            map_s       = torch.cat(memory.states_map)
            sca_s       = torch.cat(memory.states_scalars)
            actions     = torch.tensor(memory.actions)
            old_logprobs = torch.stack(memory.logprobs)

            advantages, returns = self.compute_gae(
                memory.rewards, memory.is_terminals, memory.values, 0
            )
            advantages = torch.tensor(advantages, dtype=torch.float32)
            returns    = torch.tensor(returns,    dtype=torch.float32)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(actions)

        for _ in range(PPO_EPOCHS):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, MINIBATCH_SIZE):
                end    = start + MINIBATCH_SIZE
                mb_idx = indices[start:end]

                logits, values = self.policy(map_s[mb_idx], sca_s[mb_idx])
                dist           = Categorical(logits=logits)
                new_logprobs   = dist.log_prob(actions[mb_idx])
                entropy        = dist.entropy().mean()

                ratios = torch.exp(new_logprobs - old_logprobs[mb_idx])
                mb_adv = advantages[mb_idx]
                surr1  = ratios * mb_adv
                surr2  = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                values = values.squeeze()
                mb_ret = returns[mb_idx]
                value_pred_clipped = mb_ret + (values - mb_ret).clamp(-VALUE_CLIP, VALUE_CLIP)
                critic_loss = 0.5 * torch.max(
                    (values - mb_ret).pow(2),
                    (value_pred_clipped - mb_ret).pow(2),
                ).mean()

                loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        weights = {n: p.detach().cpu().numpy() for n, p in self.policy.named_parameters()}
        np.savez(path.replace(".pth", ".npz"), **weights)


# ---------------------------------------------------------------------------
# Curriculum : choix du point de départ
# ---------------------------------------------------------------------------

def curriculum_options(ep, goal):
    """
    Retourne les options de reset adaptées à la phase du curriculum.

    Phase EASY   : starts proches du goal pour que l'agent découvre le bonus terminal.
    Phase MEDIUM : random_start activé (starts dispersés sur la carte).
    Phase HARD   : random_start + random_velocity.
    Phase FULL   : tout + wind_start_steps aléatoire.
    """
    options = {}

    if ep < PHASE_EASY:
        # Starts proches du goal (rayon ~ 20 cases) pour que l'agent
        # découvre facilement le bonus terminal et amorce l'apprentissage.
        angle  = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(8, 20)
        start  = np.array([
            np.clip(goal[0] + radius * np.cos(angle), 1, 126),
            np.clip(goal[1] + radius * np.sin(angle), 1, 126),
        ], dtype=int)
        options["start_position"] = start

    elif ep < PHASE_MEDIUM:
        options["random_start"] = True

    elif ep < PHASE_HARD:
        options["random_start"] = True
        options["wind_start_steps"] = int(np.random.randint(0, 100))

    else:
        options["random_start"]      = True
        options["random_velocity"]   = True
        options["wind_start_steps"]  = int(np.random.randint(0, 100))

    return options


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------

agent        = PPOSailingAgent(CROP_SIZE)
memory       = Memory()
history      = {s: {'rewards': [], 'collision': [], 'success': [], 'steps': [], 'shaped_rewards': []} for s in SCENARIOS}
scenario_stats = {s: {'rewards': [], 'collision': [], 'success': [], 'steps': [], 'shaped_rewards': []} for s in SCENARIOS}
total_step   = 0

for ep in tqdm(range(NUM_EPISODES)):
    scenario = SCENARIOS[ep % 3]
    env      = SailingEnv(**get_wind_scenario(scenario))

    obs, info = env.reset(options=curriculum_options(ep, env.goal_position))
    goal      = env.goal_position

    p_angle     = None
    p_dist      = np.linalg.norm(info['position'] - goal)
    ep_r        = 0
    ep_steps    = 0
    collision   = False
    milestones_reached = set()   # évite de compter plusieurs fois le même milestone

    for t in range(500):
        total_step += 1
        ep_steps   += 1

        act, lp, value, p_angle, angle_to_goal, m_in, s_in = agent.select_action(obs, goal, p_angle)
        next_obs, r, done, trunc, info = env.step(act)

        curr_dist = np.linalg.norm(info['position'] - goal)
        x, y      = info['position']

        # ------------------------------------------------------------------
        # Reward shaping
        # ------------------------------------------------------------------

        # 1. Progression vers le goal
        dist_delta = np.clip(p_dist - curr_dist, -1.0, 1.0)
        shaped_r   = r + dist_delta * DIST_DELTA_REWARD

        # 2. Pénalité temporelle (pression pour ne pas survivre passivement)
        shaped_r  += STEP_PENALTY

        # 3. Reward de cap : encourage à orienter sa vitesse vers le goal
        vel        = next_obs[2:4]
        speed      = np.linalg.norm(vel)
        if speed > 0.5:
            vel_angle  = np.arctan2(vel[1], vel[0])
            heading_err = abs(np.arctan2(
                np.sin(angle_to_goal - vel_angle),
                np.cos(angle_to_goal - vel_angle)
            ))
            # cosine similarity ∈ [-1, 1], vaut 1 si parfaitement aligné
            heading_alignment = np.cos(heading_err)
            shaped_r += HEADING_REWARD_COEF * heading_alignment

        # 4. Milestones de distance : bonus discret quand on passe sous un seuil
        #    Evite le problème de plateau où dist_delta est quasi-nul loin du goal
        for radius in [80, 60, 40, 20, 10]:
            if curr_dist < radius and radius not in milestones_reached:
                shaped_r += MILESTONE_BONUS
                milestones_reached.add(radius)

        # 5. Collision
        collision = info.get('is_stuck', False)
        if collision:
            shaped_r += COLLISION_PENALTY
            done       = True

        # Stockage
        memory.states_map.append(m_in)
        memory.states_scalars.append(s_in)
        memory.actions.append(act)
        memory.logprobs.append(lp)
        memory.rewards.append(shaped_r)
        memory.is_terminals.append(done or trunc)
        memory.values.append(value.item())

        obs    = next_obs
        ep_r  += r
        p_dist = curr_dist

        if total_step % UPDATE_TIMESTEP == 0:
            agent.update(memory)
            memory.clear()

        if done or trunc:
            break

    # ------------------------------------------------------------------
    # Fin d'épisode
    # ------------------------------------------------------------------
    agent.scheduler.step()

    success = done and not collision

    for store in [history, scenario_stats]:
        store[scenario]['rewards'].append(ep_r)
        store[scenario]['shaped_rewards'].append(shaped_r)
        store[scenario]['collision'].append(1 if collision else 0)
        store[scenario]['success'].append(1 if success else 0)
        store[scenario]['steps'].append(ep_steps if not collision else None)

    # ------------------------------------------------------------------
    # Bilan périodique
    # ------------------------------------------------------------------
    if (ep + 1) % SUMMARY_STEP_SIZE == 0:
        curr_lr = agent.optimizer.param_groups[0]['lr']
        phase   = (
            "EASY"   if ep < PHASE_EASY   else
            "MEDIUM" if ep < PHASE_MEDIUM else
            "HARD"   if ep < PHASE_HARD   else "FULL"
        )
        print(f"\n--- Bilan Eps {ep - SUMMARY_STEP_SIZE + 2}-{ep + 1} | LR: {curr_lr:.2e} | Phase: {phase} ---")

        for s in SCENARIOS:
            s_data      = scenario_stats[s]
            sr          = np.mean(s_data['success'])   * 100
            col         = np.mean(s_data['collision']) * 100
            rw          = np.mean(s_data['rewards'])
            srw         = np.mean(s_data['shaped_rewards'])
            valid_steps = [st for st in s_data['steps'] if st is not None]

            if valid_steps:
                avg_step  = np.mean(valid_steps)
                step_str  = f"{avg_step:4.1f}"
                indicator = "⭐" if avg_step < 60 else "📈" if avg_step < 90 else "🐌"
            else:
                step_str  = "--- "
                indicator = "❌"

            print(
                f"[{s:10s}] Success: {sr:3.0f}% | Collision: {col:4.1f}% | "
                f"Reward: {rw:5.1f} | Shaped: {srw:5.1f} | Steps: {step_str} {indicator}"
            )

        scenario_stats = {s: {'rewards': [], 'collision': [], 'success': [], 'steps': [], 'shaped_rewards': []} for s in SCENARIOS}

# Dernière mise à jour sur les données restantes
if len(memory.rewards) > 0:
    agent.update(memory)
    memory.clear()

agent.save("mlp_model.pth")
np.savez("history.npz", **history)