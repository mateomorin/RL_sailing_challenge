"""
PPO Sailing Agent — v3
======================
Changements majeurs vs v2 :
  1. Correction bug `forced_start` : l'option n'existait pas dans env.reset(),
     remplacé par `start_position` qui est géré nativement.
  2. Behavioral Cloning (BC) pré-entraînement : on collecte des trajectoires
     depuis un expert heuristique, puis on pré-entraîne le réseau par imitation
     avant de lancer PPO. Si tu as un agent Q-Learning entraîné, tu peux
     remplacer `heuristic_expert_action` par son policy.
  3. Curriculum learning corrigé : les phases utilisent `start_position` (géré)
     et non `forced_start` (inexistant dans env.reset).
  4. Curiosité intrinsèque légère : bonus pour visiter de nouvelles zones,
     casse les optima locaux où le bateau reste statique.
"""

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

NUM_EPISODES      = 2000
SCENARIOS         = ['training_1', 'training_2', 'training_3']
SUMMARY_STEP_SIZE = 50

# Curriculum (seuils en épisodes)
PHASE_EASY   = NUM_EPISODES // 5          # 0   – 400 : starts proches du goal
PHASE_MEDIUM = (2 * NUM_EPISODES) // 5    # 400 – 800 : random_start
PHASE_HARD   = (3 * NUM_EPISODES) // 4    # 800 – 1500 : + random_velocity
                                           # > 1500 : + wind_start_steps

# PPO
GAMMA           = 0.995
GAE_LAMBDA      = 0.95
LR              = 3e-4
UPDATE_TIMESTEP = 4096
PPO_EPOCHS      = 10
MINIBATCH_SIZE  = 256
CLIP_EPS        = 0.2
VALUE_CLIP      = 0.2
ENTROPY_COEF    = 0.03
VALUE_COEF      = 0.5
MAX_GRAD_NORM   = 0.5
CROP_SIZE       = 17

# Reward shaping
COLLISION_PENALTY = -25.0
STEP_PENALTY      = -0.02
DIST_DELTA_REWARD = 0.10
HEADING_COEF      = 0.05
MILESTONE_BONUS   = 2.0
MILESTONES        = [80, 60, 40, 20, 10]

# Curiosité intrinsèque
CURIOSITY_COEF = 0.02    # bonus par nouvelle zone visitée dans l'épisode
CURIOSITY_RES  = 16      # cases 16×16 → 8×8 zones sur la carte 128×128

# Behavioral Cloning
BC_EPISODES = 200
BC_EPOCHS   = 20
BC_LR       = 1e-3
BC_BATCH    = 256


# ---------------------------------------------------------------------------
# Architecture CNN
# ---------------------------------------------------------------------------

class SailingNet(nn.Module):
    def __init__(self, crop_size=17, n_actions=9, n_scalars=9):
        super().__init__()

        # CNN spatial : capture les relations voisinage (obstacle gauche/droite)
        # que le MLP aplati ne peut pas apprendre avec un biais inductif adéquat
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),   # (16, C, C)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=0),  # (32, C-2, C-2)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),                  # (32, 3, 3) fixe
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
# Memory
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
# Agent
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
            dx/128.0, dy/128.0,
            da/np.pi,
            obs[0]/128.0, obs[1]/128.0,
            np.sin(atg),
            dist/181.0,
        ]).unsqueeze(0)
        return s, curr_a, atg

    def select_action(self, obs, goal, prev_angle):
        with torch.no_grad():
            m_in       = self.get_local_crop(obs)
            s_in, curr_a, atg = self.build_scalars(obs, goal, prev_angle)
            logits, v  = self.policy_old(m_in, s_in)
            dist       = Categorical(logits=logits)
            action     = dist.sample()
        return action.item(), dist.log_prob(action), v.squeeze(), curr_a, atg, m_in, s_in

    def compute_gae(self, rewards, dones, values, next_value):
        adv, gae = [], 0
        vals = values + [next_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + GAMMA * vals[t+1] * (1 - dones[t]) - vals[t]
            gae   = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
            adv.insert(0, gae)
        ret = [a + v for a, v in zip(adv, vals[:-1])]
        return adv, ret

    def update(self, memory):
        with torch.no_grad():
            map_s = torch.cat(memory.states_map)
            sca_s = torch.cat(memory.states_scalars)
            acts  = torch.tensor(memory.actions)
            olp   = torch.stack(memory.logprobs)
            adv, ret = self.compute_gae(
                memory.rewards, memory.is_terminals, memory.values, 0)
            adv = torch.tensor(adv, dtype=torch.float32)
            ret = torch.tensor(ret, dtype=torch.float32)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = len(acts)
        for _ in range(PPO_EPOCHS):
            idx = np.random.permutation(N)
            for s in range(0, N, MINIBATCH_SIZE):
                mb = idx[s:s+MINIBATCH_SIZE]
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
                loss = aloss + VALUE_COEF * closs - ENTROPY_COEF * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    # ------------------------------------------------------------------
    # Behavioral Cloning
    # ------------------------------------------------------------------

    def behavioral_cloning(self, demos):
        """
        Pré-entraîne le réseau par imitation sur des trajectoires d'expert.
        demos : liste de (map_tensor 1×3×C×C, scalar_tensor 1×9, action_int)

        Pour utiliser ton propre agent Q-Learning à la place de l'heuristique,
        remplace `heuristic_expert_action` dans `collect_expert_demos` par
        ton policy Q-Learning : action = q_agent.act(obs)
        """
        print(f"\n=== Behavioral Cloning — {len(demos)} transitions ===")
        opt = optim.Adam(self.policy.parameters(), lr=BC_LR)

        maps    = torch.cat([d[0] for d in demos])
        scalars = torch.cat([d[1] for d in demos])
        actions = torch.tensor([d[2] for d in demos])

        N = len(actions)
        for epoch in range(BC_EPOCHS):
            idx        = np.random.permutation(N)
            total_loss = 0.0
            for s in range(0, N, BC_BATCH):
                mb      = idx[s:s+BC_BATCH]
                logits, _ = self.policy(maps[mb], scalars[mb])
                loss    = F.cross_entropy(logits, actions[mb])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), MAX_GRAD_NORM)
                opt.step()
                total_loss += loss.item() * len(mb)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{BC_EPOCHS} | Loss: {total_loss/N:.4f}")

        self.policy_old.load_state_dict(self.policy.state_dict())
        print("=== BC terminé ===\n")

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        weights = {n: p.detach().cpu().numpy() for n, p in self.policy.named_parameters()}
        np.savez(path.replace(".pth", ".npz"), **weights)


# ---------------------------------------------------------------------------
# Expert heuristique
# ---------------------------------------------------------------------------

def heuristic_expert_action(obs, goal, world_map):
    """
    Expert simple : se dirige vers le goal en évitant les murs immédiats.
    Utilisé pour la collecte de démos BC.

    REMPLACER PAR TON Q-LEARNING si disponible :
        action = q_agent.act(obs)   # ton agent entraîné
    """
    pos = obs[0:2].astype(int)
    directions = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    best_a, best_d = 8, float('inf')
    for i, (dx, dy) in enumerate(directions):
        nx = int(np.clip(pos[0] + dx, 0, 127))
        ny = int(np.clip(pos[1] + dy, 0, 127))
        if world_map[ny, nx] == 1:
            continue
        d = np.linalg.norm(np.array([nx, ny]) - goal)
        if d < best_d:
            best_d, best_a = d, i
    return best_a


def collect_expert_demos(agent, n_episodes=BC_EPISODES):
    """
    Collecte des trajectoires d'expert (succès uniquement) pour le BC.
    Lance des épisodes avec starts proches du goal pour maximiser
    le taux de succès de l'heuristique simple.
    """
    demos, success = [], 0
    print(f"\n=== Collecte démos expert ({n_episodes} épisodes) ===")

    for ep in tqdm(range(n_episodes)):
        scenario = SCENARIOS[ep % 3]
        env      = SailingEnv(**get_wind_scenario(scenario))
        goal     = env.goal_position
        wmap     = env._create_world()

        # Start proche du goal (l'heuristique glouton réussit souvent sur <30 cases)
        start_opts = {}
        for _ in range(30):
            a  = np.random.uniform(0, 2 * np.pi)
            r  = np.random.uniform(5, 30)
            sx = int(np.clip(goal[0] + r * np.cos(a), 1, 126))
            sy = int(np.clip(goal[1] + r * np.sin(a), 1, 126))
            if wmap[sy, sx] == 0:
                start_opts = {"start_position": [sx, sy]}
                break

        obs, info = env.reset(options=start_opts)
        goal      = env.goal_position
        prev_a    = None
        ep_demos  = []

        for _ in range(500):
            act      = heuristic_expert_action(obs, goal, wmap)
            m_in     = agent.get_local_crop(obs)
            s_in, prev_a, _ = agent.build_scalars(obs, goal, prev_a)
            ep_demos.append((m_in, s_in, act))

            obs, r, done, trunc, info = env.step(act)
            if done or trunc:
                if r > 0:
                    demos.extend(ep_demos)
                    success += 1
                break

    print(f"  {len(demos)} transitions collectées ({success}/{n_episodes} succès)")
    return demos


# ---------------------------------------------------------------------------
# Curiosité intrinsèque
# ---------------------------------------------------------------------------

class CuriosityTracker:
    """Bonus pour la première visite d'une zone 16×16 dans l'épisode."""
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
# Curriculum
# ---------------------------------------------------------------------------

def curriculum_options(ep, goal, world_map):
    """
    Options de reset selon la phase.
    Utilise 'start_position' (géré par env.reset) et non 'forced_start'.
    """
    options = {}

    if ep < PHASE_EASY:
        for _ in range(50):
            a  = np.random.uniform(0, 2 * np.pi)
            r  = np.random.uniform(8, 28)
            sx = int(np.clip(goal[0] + r * np.cos(a), 1, 126))
            sy = int(np.clip(goal[1] + r * np.sin(a), 1, 126))
            if world_map[sy, sx] == 0:
                options["start_position"] = [sx, sy]
                break

    elif ep < PHASE_MEDIUM:
        options["random_start"] = True

    elif ep < PHASE_HARD:
        options["random_start"]     = True
        options["wind_start_steps"] = int(np.random.randint(0, 100))

    else:
        options["random_start"]    = True
        options["random_velocity"] = True
        options["wind_start_steps"] = int(np.random.randint(0, 100))

    return options


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

agent     = PPOSailingAgent(CROP_SIZE)
memory    = Memory()
curiosity = CuriosityTracker()

# Phase 0 : Behavioral Cloning
demos = collect_expert_demos(agent, BC_EPISODES)
if demos:
    print("\n=== Pré-entraînement BC ===")
    agent.behavioral_cloning(demos)
del demos

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

    obs, info = env.reset(options=curriculum_options(ep, goal, wmap))
    goal      = env.goal_position

    curiosity.reset()
    p_angle   = None
    p_dist    = np.linalg.norm(info['position'] - goal)
    ep_r      = 0
    ep_steps  = 0
    collision = False
    milestones = set()

    for t in range(500):
        total_step += 1
        ep_steps   += 1

        act, lp, value, p_angle, atg, m_in, s_in = agent.select_action(
            obs, goal, p_angle)
        next_obs, r, done, trunc, info = env.step(act)

        curr_dist = np.linalg.norm(info['position'] - goal)
        collision = info.get('is_stuck', False)

        # Reward shaping
        dist_delta = np.clip(p_dist - curr_dist, -1.0, 1.0)
        shaped_r   = r + dist_delta * DIST_DELTA_REWARD + STEP_PENALTY

        vel   = next_obs[2:4]
        speed = np.linalg.norm(vel)
        if speed > 0.5:
            vel_a    = np.arctan2(vel[1], vel[0])
            err      = abs(np.arctan2(np.sin(atg - vel_a), np.cos(atg - vel_a)))
            shaped_r += HEADING_COEF * np.cos(err)

        for radius in MILESTONES:
            if curr_dist < radius and radius not in milestones:
                shaped_r += MILESTONE_BONUS
                milestones.add(radius)

        shaped_r += curiosity.bonus(info['position'])

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

    if (ep + 1) % SUMMARY_STEP_SIZE == 0:
        lr    = agent.optimizer.param_groups[0]['lr']
        phase = ("EASY"   if ep < PHASE_EASY   else
                 "MEDIUM" if ep < PHASE_MEDIUM else
                 "HARD"   if ep < PHASE_HARD   else "FULL")
        print(f"\n--- Ep {ep+1} | LR: {lr:.2e} | Phase: {phase} ---")
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

agent.save("mlp_model.pth")
np.savez("history.npz", **history)