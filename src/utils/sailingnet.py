"""
PPO Sailing Agent — v4
======================

Philosophie de cette version
------------------------------
Le point de départ est FIXE et CONNU : [64, 0], vitesse nulle.
C'est aussi la configuration du test. On en tire parti plutôt que de
l'ignorer avec un curriculum aléatoire qui cause du catastrophic forgetting.

Pipeline :
  1. BC    — collecte des démos windmaster DEPUIS LE POINT FIXE,
             pré-entraîne le réseau sur ces trajectoires.
  2. PPO   — affine depuis ce même point fixe (+ léger bruit de position
             après convergence initiale, pas de randomisation agressive).

Pourquoi pas de curriculum agressif :
  - Le point de départ est fixe au test → apprendre depuis des starts
    aléatoires crée un mismatch train/test.
  - Phases EASY→MEDIUM→HARD causent du catastrophic forgetting :
    l'agent oublie la politique EASY quand il voit des starts lointains.
  - Windmaster résout le problème depuis le point fixe → ses démos
    couvrent déjà les trajectoires difficiles autour de l'île.

Pourquoi BC sur windmaster fonctionne ici (vs avant) :
  - On collecte les démos DEPUIS [64,0], pas depuis des starts aléatoires
    proches du goal → le réseau apprend exactement les transitions du problème réel.
  - On garde uniquement les épisodes RÉUSSIS de windmaster.
  - On ajoute un replay buffer des démos BC pendant PPO (DAgger-lite) pour
    éviter l'oubli catastrophique du comportement expert.
"""

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import random
from collections import deque

from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
from src.agents.windmaster import MyAgent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS   = ['training_1', 'training_2', 'training_3']
SUMMARY_N   = 50

# Point de départ fixe (identique à l'environnement de test)
DEFAULT_START = [64, 0]

# PPO
NUM_EPISODES    = 2000
GAMMA           = 0.995
GAE_LAMBDA      = 0.95
LR              = 5e-5
UPDATE_TIMESTEP = 4096
PPO_EPOCHS      = 3
MINIBATCH_SIZE  = 256
CLIP_EPS        = 0.2
VALUE_CLIP      = 0.2
ENTROPY_COEF    = 0.001
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
CROP_SIZE       = 17

# Reward shaping
COLLISION_PENALTY = -10.0
STEP_PENALTY      = -0.1
DIST_DELTA_REWARD = 0.50
HEADING_COEF      = 0.05
MILESTONE_BONUS   = 2.0
MILESTONES        = [100, 80, 60, 40, 20, 10]

# Curiosité intrinsèque (anti-stagnation)
CURIOSITY_COEF = 0.001
CURIOSITY_RES  = 16

# Behavioral Cloning
BC_EPISODES     = 300   # épisodes windmaster à collecter
BC_EPOCHS       = 30
BC_LR           = 1e-3
BC_BATCH        = 256

# DAgger-lite : replay des démos BC pendant PPO pour éviter l'oubli
DEMO_REPLAY_FRAC  = 0.5   # 50% de chaque batch PPO vient des démos BC
DEMO_BUFFER_SIZE  = 50_000  # transitions BC conservées en mémoire

# Légère perturbation de position après convergence initiale
# (pas de curriculum agressif — juste un bruit de ±5 cases après ep 500)
PERTURB_START_EP  = 500
PERTURB_RADIUS    = 5   # cases


# ---------------------------------------------------------------------------
# Architecture CNN
# ---------------------------------------------------------------------------

class SailingNet(nn.Module):
    """
    CNN pour le crop local + MLP pour les scalaires.
    Le CNN capture les relations spatiales (obstacle gauche/droite/devant)
    que le MLP aplati ne peut pas apprendre avec un biais inductif adéquat.
    """

    def __init__(self, crop_size=17, n_actions=9, n_scalars=9):
        super().__init__()

        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # → (16, C, C)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),  # → (32, C-2, C-2)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                  # → (32, 3, 3) fixe
            nn.Flatten(),
            nn.Linear(288, 64),
            nn.ReLU(),
        )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(n_scalars, 48),
            nn.Tanh(),
        )

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
        x = torch.cat([self.map_encoder(local_map),
                        self.scalar_encoder(scalars)], dim=1)
        x = self.shared(x)
        return self.actor(x), self.critic(x)


# ---------------------------------------------------------------------------
# Memory PPO
# ---------------------------------------------------------------------------

class Memory:
    def __init__(self):
        self.states_map     = []
        self.states_scalars = []
        self.actions        = []
        self.logprobs       = []
        self.rewards        = []
        self.is_terminals   = []
        self.values         = []

    def clear(self):
        for attr in vars(self):
            getattr(self, attr).clear()


# ---------------------------------------------------------------------------
# Agent PPO
# ---------------------------------------------------------------------------

class PPOSailingAgent:

    def __init__(self, crop_size=CROP_SIZE):
        self.crop_size  = crop_size
        self.policy     = SailingNet(crop_size=crop_size)
        self.optimizer  = optim.Adam(self.policy.parameters(), lr=LR, eps=1e-5)
        self.scheduler  = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_EPISODES)
        self.policy_old = SailingNet(crop_size=crop_size)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Buffer de démos BC pour DAgger-lite (évite catastrophic forgetting)
        self.demo_buffer = deque(maxlen=DEMO_BUFFER_SIZE)

    # ------------------------------------------------------------------
    # Extraction features
    # ------------------------------------------------------------------

    def get_local_crop(self, obs):
        pos    = obs[0:2].astype(int)
        wmap   = obs[32774:49158].reshape(128, 128)
        wfield = obs[6:32774].reshape(128, 128, 2)
        pad    = self.crop_size // 2

        wmap_p = np.pad(wmap,   pad, constant_values=1)
        wf_p   = np.pad(wfield, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)

        y, x  = pos[1] + pad, pos[0] + pad
        crop  = np.zeros((3, self.crop_size, self.crop_size), dtype=np.float32)
        crop[0]  = wmap_p [y-pad:y+pad+1, x-pad:x+pad+1]
        crop[1:] = wf_p   [y-pad:y+pad+1, x-pad:x+pad+1].transpose(2, 0, 1)
        return torch.FloatTensor(crop).unsqueeze(0)

    def build_scalars(self, obs, goal, prev_angle):
        curr_w = obs[4:6]
        curr_a = np.arctan2(curr_w[1], curr_w[0])
        da     = 0.0 if prev_angle is None else np.arctan2(
            np.sin(curr_a - prev_angle), np.cos(curr_a - prev_angle))
        dx, dy = goal[0] - obs[0], goal[1] - obs[1]
        dist   = np.sqrt(dx**2 + dy**2)
        atg    = np.arctan2(dy, dx)
        s = torch.FloatTensor([
            obs[2]/10.0, obs[3]/10.0,
            dx/128.0,    dy/128.0,
            da/np.pi,
            obs[0]/128.0, obs[1]/128.0,
            np.sin(atg),
            dist/181.0,
        ]).unsqueeze(0)
        return s, curr_a, atg

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------

    def select_action(self, obs, goal, prev_angle):
        with torch.no_grad():
            m_in          = self.get_local_crop(obs)
            s_in, ca, atg = self.build_scalars(obs, goal, prev_angle)
            logits, v     = self.policy_old(m_in, s_in)
            dist          = Categorical(logits=logits)
            action        = dist.sample()
        return action.item(), dist.log_prob(action), v.squeeze(), ca, atg, m_in, s_in

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def compute_gae(self, rewards, dones, values, next_val=0):
        adv, gae = [], 0
        vals = values + [next_val]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * vals[t+1] * (1 - dones[t]) - vals[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            adv.insert(0, gae)
        ret = [a + v for a, v in zip(adv, vals[:-1])]
        return adv, ret

    # ------------------------------------------------------------------
    # PPO update avec DAgger-lite
    # ------------------------------------------------------------------

    def update(self, memory):
        with torch.no_grad():
            map_s = torch.cat(memory.states_map)
            sca_s = torch.cat(memory.states_scalars)
            acts  = torch.tensor(memory.actions)
            olp   = torch.stack(memory.logprobs)
            adv, ret = self.compute_gae(
                memory.rewards, memory.is_terminals, memory.values)
            adv = torch.tensor(adv, dtype=torch.float32)
            ret = torch.tensor(ret, dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = len(acts)
        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(N)

            for s in range(0, N, MINIBATCH_SIZE):
                mb = idx[s:s+MINIBATCH_SIZE]

                # --- PPO classique ---
                logits, vals = self.policy(map_s[mb], sca_s[mb])
                dist    = Categorical(logits=logits)
                nlp     = dist.log_prob(acts[mb])
                entropy = dist.entropy().mean()
                ratios  = torch.exp(nlp - olp[mb])
                mb_adv  = adv[mb]
                aloss   = -torch.min(
                    ratios * mb_adv,
                    torch.clamp(ratios, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv
                ).mean()
                v      = vals.squeeze()
                mb_ret = ret[mb]
                vc     = mb_ret + (v - mb_ret).clamp(-VALUE_CLIP, VALUE_CLIP)
                closs  = 0.5 * torch.max(
                    (v - mb_ret).pow(2), (vc - mb_ret).pow(2)).mean()

                # --- DAgger-lite : régularisation BC sur les démos expert ---
                # Empêche d'oublier les comportements windmaster pendant PPO.
                bc_loss = torch.tensor(0.0)
                if len(self.demo_buffer) > BC_BATCH:
                    n_demo = max(1, int(len(mb) * DEMO_REPLAY_FRAC))
                    demo_batch = random.sample(self.demo_buffer, n_demo)
                    d_maps    = torch.cat([d[0] for d in demo_batch])
                    d_scalars = torch.cat([d[1] for d in demo_batch])
                    d_acts    = torch.tensor([d[2] for d in demo_batch])
                    d_logits, _ = self.policy(d_maps, d_scalars)
                    bc_loss = F.cross_entropy(d_logits, d_acts)

                loss = aloss + VALUE_COEF * closs - ENTROPY_COEF * entropy + 1 * bc_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    # Behavioral Cloning (phase initiale)
    # ------------------------------------------------------------------

    def behavioral_cloning(self, demos):
        """
        Pré-entraîne le réseau sur les démos windmaster par cross-entropy.
        Stocke aussi les démos dans le buffer DAgger pour l'entraînement PPO.
        """
        print(f"\n=== Behavioral Cloning — {len(demos)} transitions ===")
        opt = optim.Adam(self.policy.parameters(), lr=BC_LR)

        # Stocker dans le replay buffer DAgger
        self.demo_buffer.extend(demos)

        maps    = torch.cat([d[0] for d in demos])
        scalars = torch.cat([d[1] for d in demos])
        actions = torch.tensor([d[2] for d in demos])

        N = len(actions)
        best_loss = float('inf')
        for epoch in range(BC_EPOCHS):
            idx = np.random.permutation(N)
            total_loss = 0.0
            for s in range(0, N, BC_BATCH):
                mb = idx[s:s+BC_BATCH]
                logits, _ = self.policy(maps[mb], scalars[mb])
                loss = F.cross_entropy(logits, actions[mb])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                opt.step()
                total_loss += loss.item() * len(mb)
            avg = total_loss / N
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{BC_EPOCHS} | Loss: {avg:.4f}")
            if avg < best_loss:
                best_loss = avg
                # Sauvegarde du meilleur état BC
                self._best_bc_state = {k: v.clone() for k, v in self.policy.state_dict().items()}

        # Charge le meilleur état plutôt que le dernier
        if hasattr(self, '_best_bc_state'):
            self.policy.load_state_dict(self._best_bc_state)
        self.policy_old.load_state_dict(self.policy.state_dict())
        print(f"  Meilleure loss BC : {best_loss:.4f}\n=== BC terminé ===\n")

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        weights = {n: p.detach().cpu().numpy() for n, p in self.policy.named_parameters()}
        np.savez(path.replace(".pth", ".npz"), **weights)


# ---------------------------------------------------------------------------
# Collecte des démos windmaster
# ---------------------------------------------------------------------------

def collect_windmaster_demos(agent, n_episodes=BC_EPISODES):
    """
    Roule windmaster DEPUIS LE POINT DE DÉPART FIXE [64, 0].
    Ne conserve que les épisodes réussis.

    Pourquoi le point fixe uniquement :
      - C'est la config du test → les démos couvrent exactement le problème réel.
      - Windmaster réussit souvent depuis ce point → démos de qualité.
      - Pas de mismatch entre les démos et les épisodes PPO.
    """
    if MyAgent is None:
        print("AVERTISSEMENT: windmaster non trouvé, BC ignoré.")
        return []

    expert   = MyAgent()
    demos    = []
    success  = 0
    failed   = 0

    print(f"\n=== Collecte démos windmaster ({n_episodes} épisodes depuis {DEFAULT_START}) ===")

    for ep in tqdm(range(n_episodes)):
        scenario = SCENARIOS[ep % 3]
        env      = SailingEnv(**get_wind_scenario(scenario))

        # Toujours depuis le point de départ fixe, vitesse nulle
        obs, info = env.reset(options={"start_position": DEFAULT_START})
        expert.reset()
        goal     = env.goal_position

        ep_demos  = []
        prev_a    = None
        ep_success = False

        for _ in range(500):
            # Action windmaster
            act  = expert.act(obs)

            # Features pour notre réseau
            m_in          = agent.get_local_crop(obs)
            s_in, prev_a, _ = agent.build_scalars(obs, goal, prev_a)
            ep_demos.append((m_in, s_in, act))

            obs, r, done, trunc, info = env.step(act)
            if done or trunc:
                if r > 0:
                    ep_success = True
                break

        if ep_success:
            demos.extend(ep_demos)
            success += 1
        else:
            failed += 1

    print(f"  Succès : {success}/{n_episodes} | "
          f"Transitions conservées : {len(demos)} | "
          f"Taux : {success/n_episodes*100:.1f}%")

    if success == 0:
        print("  AVERTISSEMENT: windmaster n'a pas réussi depuis [64,0]. "
              "Vérifier la configuration de l'environnement.")

    return demos


# ---------------------------------------------------------------------------
# Curiosité intrinsèque
# ---------------------------------------------------------------------------

class CuriosityTracker:
    """Bonus pour première visite d'une zone 16×16 dans l'épisode."""
    def __init__(self, resolution=CURIOSITY_RES):
        self.res     = resolution
        self.visited = set()

    def reset(self):
        self.visited = set()

    def bonus(self, pos):
        cell = (int(pos[0]) // self.res, int(pos[1]) // self.res)
        if cell not in self.visited:
            self.visited.add(cell)
            return CURIOSITY_COEF
        return 0.0


# ---------------------------------------------------------------------------
# Options de reset
# ---------------------------------------------------------------------------

def get_reset_options(ep, world_map):
    """
    Pas de curriculum agressif.

    Épisodes 0 – PERTURB_START_EP : toujours le point fixe [64, 0].
    Épisodes > PERTURB_START_EP   : point fixe + bruit ±PERTURB_RADIUS cases.

    Pourquoi pas de random_start :
      - Le test utilise toujours [64, 0].
      - random_start crée un mismatch train/test et du catastrophic forgetting.
      - windmaster réussit depuis [64, 0] → on veut que PPO apprenne depuis là.
    """
    if ep < PERTURB_START_EP:
        return {"start_position": DEFAULT_START}

    # Légère perturbation après convergence initiale
    for _ in range(20):
        noise_x = int(np.random.randint(-PERTURB_RADIUS, PERTURB_RADIUS + 1))
        noise_y = int(np.random.randint(-PERTURB_RADIUS, PERTURB_RADIUS + 1))
        sx = int(np.clip(DEFAULT_START[0] + noise_x, 1, 126))
        sy = int(np.clip(DEFAULT_START[1] + noise_y, 1, 126))
        if world_map[sy, sx] == 0:
            return {"start_position": [sx, sy]}

    return {"start_position": DEFAULT_START}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

agent     = PPOSailingAgent(CROP_SIZE)
memory    = Memory()
curiosity = CuriosityTracker()

# --- Phase 1 : Behavioral Cloning depuis windmaster ---
demos = collect_windmaster_demos(agent, BC_EPISODES)
if demos:
    agent.behavioral_cloning(demos)
    # Évaluation rapide post-BC
    print("=== Évaluation post-BC (10 épisodes depuis [64,0]) ===")
    bc_successes = 0
    for ep in range(10):
        env_eval = SailingEnv(**get_wind_scenario(SCENARIOS[ep % 3]))
        obs_e, _ = env_eval.reset(options={"start_position": DEFAULT_START})
        goal_e   = env_eval.goal_position
        pa_e     = None
        for _ in range(500):
            act_e, _, _, pa_e, _, _, _ = agent.select_action(obs_e, goal_e, pa_e)
            obs_e, r_e, done_e, trunc_e, _ = env_eval.step(act_e)
            if done_e or trunc_e:
                if r_e > 0: bc_successes += 1
                break
    print(f"  Succès post-BC : {bc_successes}/10\n")
del demos

# --- Phase 2 : PPO depuis le point fixe ---
history        = {s: {'rewards': [], 'collision': [], 'success': [],
                       'steps': [], 'shaped_rewards': []} for s in SCENARIOS}
scenario_stats = {s: {'rewards': [], 'collision': [], 'success': [],
                       'steps': [], 'shaped_rewards': []} for s in SCENARIOS}
total_step = 0

for ep in tqdm(range(NUM_EPISODES)):
    scenario = SCENARIOS[ep % 3]
    env      = SailingEnv(**get_wind_scenario(scenario))
    goal     = env.goal_position
    wmap     = env._create_world()

    obs, info = env.reset(options=get_reset_options(ep, wmap))
    goal      = env.goal_position

    curiosity.reset()
    p_angle   = None
    p_dist    = np.linalg.norm(info['position'] - goal)
    ep_r      = 0
    ep_steps  = 0
    collision = False
    milestones = set()
    shaped_r  = 0.0

    for t in range(500):
        total_step += 1
        ep_steps   += 1

        act, lp, value, p_angle, atg, m_in, s_in = agent.select_action(obs, goal, p_angle)
        next_obs, r, done, trunc, info = env.step(act)

        curr_dist = np.linalg.norm(info['position'] - goal)
        collision = info.get('is_stuck', False)

        # Reward shaping
        dist_delta = np.clip(p_dist - curr_dist, -1.0, 1.0)
        shaped_r   = r + dist_delta * DIST_DELTA_REWARD + STEP_PENALTY

        # Cap vers le goal
        vel   = next_obs[2:4]
        speed = np.linalg.norm(vel)
        if speed > 0.5:
            vel_a    = np.arctan2(vel[1], vel[0])
            err      = abs(np.arctan2(np.sin(atg - vel_a), np.cos(atg - vel_a)))
            shaped_r += HEADING_COEF * np.cos(err)

        # Milestones de distance
        for radius in MILESTONES:
            if curr_dist < radius and radius not in milestones:
                shaped_r += MILESTONE_BONUS
                milestones.add(radius)

        # Curiosité
        shaped_r += curiosity.bonus(info['position'])

        # Collision
        if collision:
            shaped_r += COLLISION_PENALTY
            done       = True

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

    agent.scheduler.step()
    success = done and not collision

    for store in [history, scenario_stats]:
        store[scenario]['rewards'].append(ep_r)
        store[scenario]['shaped_rewards'].append(shaped_r)
        store[scenario]['collision'].append(1 if collision else 0)
        store[scenario]['success'].append(1 if success else 0)
        store[scenario]['steps'].append(ep_steps if not collision else None)

    if (ep + 1) % SUMMARY_N == 0:
        lr    = agent.optimizer.param_groups[0]['lr']
        perturb = "PERTURB" if ep >= PERTURB_START_EP else "FIXED"
        print(f"\n--- Ep {ep+1}/{NUM_EPISODES} | LR: {lr:.2e} | Start: {perturb} ---")
        for s in SCENARIOS:
            d   = scenario_stats[s]
            sr  = np.mean(d['success'])   * 100
            col = np.mean(d['collision']) * 100
            rw  = np.mean(d['rewards'])
            srw = np.mean(d['shaped_rewards'])
            vs  = [x for x in d['steps'] if x is not None]
            step_str  = f"{np.mean(vs):4.1f}" if vs else "--- "
            indicator = ("⭐" if vs and np.mean(vs) < 60 else
                         "📈" if vs and np.mean(vs) < 90 else
                         "🐌" if vs else "❌")
            print(f"  [{s:10s}] Succ: {sr:3.0f}% | Coll: {col:4.1f}% | "
                  f"Rwd: {rw:5.1f} | Shaped: {srw:5.1f} | Steps: {step_str} {indicator}")
        scenario_stats = {s: {'rewards': [], 'collision': [], 'success': [],
                               'steps': [], 'shaped_rewards': []} for s in SCENARIOS}

if memory.rewards:
    agent.update(memory)
    memory.clear()

agent.save("no_curriculum_mlp_model.pth")
np.savez("no_curriculum_history.npz", **history)
print("\nEntraînement terminé. Modèle sauvegardé : no_curriculum_mlp_model.pth")