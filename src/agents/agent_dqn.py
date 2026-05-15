"""
DQN Agent avec réseau neuronal profond — Sailing Challenge
==========================================================

Architecture :
  - Extraction de ~46 features compactes depuis l'observation brute
    (identique à agent_ppo.py : position, vitesse, vent local, champ de vent
     par zone, prédiction t+1, distances île/goal, efficacités directionnelles)
  - Réseau Q  [46 → 256 → 256 → 128 → 9]  avec Dueling DQN
  - Double DQN (réseau cible mis à jour par soft update)
  - Prioritized Experience Replay (PER) avec importance sampling
  - Epsilon-greedy avec decay exponentiel
  - Shaped reward intermédiaire (progression + pénalité île + bonus vent)
  - Inférence numpy-only pour la soumission Codabench

Utilisation :
  python agent_dqn.py --train           # entraîne et sauvegarde
  python agent_dqn.py --eval            # évalue le modèle sauvegardé
  python agent_dqn.py --train --eval    # les deux

Dépendances entraînement : torch, numpy, gymnasium
Dépendances inférence    : numpy uniquement
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (numpy, partagé entraînement + inférence)
# ═══════════════════════════════════════════════════════════════════════════════

GRID_SIZE = 128
GOAL_X    = 64
GOAL_Y    = 127

# Géométrie de l'île
ISLAND_RECT  = (45, 83, 45, 83)   # x_min, x_max, y_min, y_max
ISLAND_TIP_Y = 22


def _wind_at(wind_field_2d, x, y):
    xi = int(np.clip(x, 0, GRID_SIZE - 1))
    yi = int(np.clip(y, 0, GRID_SIZE - 1))
    return wind_field_2d[yi, xi]


def _predict_next_wind(wind_field_2d, mean_rot_deg=3.0):
    theta  = np.deg2rad(mean_rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = wind_field_2d[:, :, 0]
    y = wind_field_2d[:, :, 1]
    return np.stack([x * cos_t - y * sin_t, x * sin_t + y * cos_t], axis=-1)


def _zone_mean_wind(wind_field_2d, x_min, x_max, y_min, y_max):
    x_min = int(np.clip(x_min, 0, GRID_SIZE - 1))
    x_max = int(np.clip(x_max, 0, GRID_SIZE - 1)) + 1
    y_min = int(np.clip(y_min, 0, GRID_SIZE - 1))
    y_max = int(np.clip(y_max, 0, GRID_SIZE - 1)) + 1
    zone  = wind_field_2d[y_min:y_max, x_min:x_max]
    if zone.size == 0:
        return np.zeros(2)
    return zone.reshape(-1, 2).mean(axis=0)


def _sailing_efficiency(boat_dir, wind_dir):
    w_norm = np.linalg.norm(wind_dir)
    b_norm = np.linalg.norm(boat_dir)
    if w_norm < 1e-9 or b_norm < 1e-9:
        return 0.05
    wind_from = -wind_dir / w_norm
    b = boat_dir / b_norm
    cos_a = np.clip(np.dot(wind_from, b), -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < np.pi / 4:
        return 0.05
    elif angle < np.pi / 2:
        return 0.5 + 0.5 * (angle - np.pi / 4) / (np.pi / 4)
    elif angle < 3 * np.pi / 4:
        return 1.0
    else:
        return max(0.5, 1.0 - 0.5 * (angle - 3 * np.pi / 4) / (np.pi / 4))


def extract_features(obs: np.ndarray) -> np.ndarray:
    """
    Transforme l'observation brute (~65K floats) en vecteur de 46 features.

    [0-1]   position normalisée
    [2-3]   vitesse (vx, vy)
    [4-5]   vent local normalisé
    [6]     efficacité de voile actuelle
    [7-9]   direction + distance vers le goal
    [10-13] distances aux 4 bords de l'île
    [14-15] indicateurs route NORD / SUD
    [16-19] vent moyen couloirs NORD et SUD normalisés
    [20-21] vent moyen zone goal normalisé
    [22-23] vent prédit t+1 à position actuelle normalisé
    [24-29] efficacités pour 6 directions clés vs vent local
    [30-31] angle et magnitude vent local
    [32]    placeholder (step progress)
    [33-38] efficacités de voile sur 6 waypoints vers le goal
    [39-42] angle vent moyen des 4 quadrants (détection asymétrie)
    [43]    avantage route NORD vs SUD
    [44]    magnitude vitesse normalisée
    [45]    indicateur proximité île (<10 cells)
    """
    x,  y  = float(obs[0]),  float(obs[1])
    vx, vy = float(obs[2]),  float(obs[3])
    wx, wy = float(obs[4]),  float(obs[5])

    wind_flat  = obs[6 : 6 + GRID_SIZE * GRID_SIZE * 2]
    wind_field = wind_flat.reshape(GRID_SIZE, GRID_SIZE, 2)

    wind_speed = np.sqrt(wx**2 + wy**2) + 1e-9
    feat = []

    feat.extend([x / GRID_SIZE, y / GRID_SIZE])
    feat.extend([vx, vy])
    feat.extend([wx / wind_speed, wy / wind_speed])

    v_speed = np.sqrt(vx**2 + vy**2)
    eff_cur = _sailing_efficiency(np.array([vx, vy]), np.array([wx, wy])) if v_speed > 0.1 else 0.0
    feat.append(eff_cur)

    gx, gy    = GOAL_X - x, GOAL_Y - y
    dist_goal = np.sqrt(gx**2 + gy**2) + 1e-9
    feat.extend([gx / dist_goal, gy / dist_goal, dist_goal / GRID_SIZE])

    xl, xr, yb, yt = ISLAND_RECT
    feat.extend([
        (yb - y) / GRID_SIZE,
        (y  - yt) / GRID_SIZE,
        (x  - xl) / GRID_SIZE,
        (xr - x)  / GRID_SIZE,
    ])

    feat.extend([float(y > yt), float(y < yb)])

    w_north = _zone_mean_wind(wind_field, 0, 127, yt + 1, 127)
    w_south = _zone_mean_wind(wind_field, 0, 127, 0, yb - 1)
    wns = np.linalg.norm(w_north) + 1e-9
    wss = np.linalg.norm(w_south) + 1e-9
    feat.extend([w_north[0]/wns, w_north[1]/wns, w_south[0]/wss, w_south[1]/wss])

    w_goal  = _zone_mean_wind(wind_field, 44, 84, yt + 1, 127)
    wgs     = np.linalg.norm(w_goal) + 1e-9
    feat.extend([w_goal[0]/wgs, w_goal[1]/wgs])

    wind_next = _predict_next_wind(wind_field)
    wn        = _wind_at(wind_next, x, y)
    wns2      = np.linalg.norm(wn) + 1e-9
    feat.extend([wn[0]/wns2, wn[1]/wns2])

    for dx, dy in [(0,1),(1,1),(1,0),(0,-1),(-1,0),(-1,1)]:
        feat.append(_sailing_efficiency(np.array([dx,dy]), np.array([wx,wy])))

    feat.extend([np.arctan2(wy, wx) / np.pi, wind_speed / 10.0, 0.0])

    for i in range(1, 7):
        t  = i / 7.0
        px = x + t * gx
        py = y + t * gy
        wp = _wind_at(wind_field, px, py)
        feat.append(_sailing_efficiency(np.array([gx, gy]), wp))

    mid = GRID_SIZE // 2
    for (x0,x1,y0,y1) in [(0,mid,mid,127),(mid,127,mid,127),(0,mid,0,mid),(mid,127,0,mid)]:
        wq = _zone_mean_wind(wind_field, x0, x1, y0, y1)
        feat.append(np.arctan2(wq[1], wq[0]) / np.pi)

    eff_n = _sailing_efficiency(np.array([0.0, 1.0]), w_north)
    eff_s = _sailing_efficiency(np.array([0.0,-1.0]), w_south)
    feat.append(eff_n - eff_s)
    feat.append(v_speed / 8.0)

    dist_to_island = min(abs(x-xl), abs(x-xr), abs(y-yb), abs(y-yt))
    feat.append(float(dist_to_island < 10))

    return np.array(feat, dtype=np.float32)


N_FEATURES = 46
N_ACTIONS  = 9


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAPED REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def shaped_reward(obs_prev, obs_curr, env_reward, is_stuck):
    x_p, y_p = obs_prev[0], obs_prev[1]
    x_c, y_c = obs_curr[0], obs_curr[1]
    wx,  wy  = obs_curr[4], obs_curr[5]
    vx,  vy  = obs_curr[2], obs_curr[3]

    d_prev    = np.sqrt((GOAL_X - x_p)**2 + (GOAL_Y - y_p)**2)
    d_curr    = np.sqrt((GOAL_X - x_c)**2 + (GOAL_Y - y_c)**2)
    progress  = (d_prev - d_curr) / GRID_SIZE * 5.0

    collision = -50.0 if is_stuck else 0.0
    time_pen  = -0.02

    v_speed   = np.sqrt(vx**2 + vy**2)
    eff_bonus = 0.0
    if v_speed > 0.1:
        eff_bonus = _sailing_efficiency(np.array([vx,vy]), np.array([wx,wy])) * 0.05

    total = env_reward + progress + collision + time_pen + eff_bonus
    return float(total), {
        'progress':           float(progress),
        'collision_penalty':  float(collision),
        'time_penalty':       float(time_pen),
        'eff_bonus':          float(eff_bonus),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIORITIZED EXPERIENCE REPLAY
# ═══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay avec Sum-Tree pour un sampling O(log N).
    alpha : exposant de priorité (0 = uniforme, 1 = full priority)
    beta  : exposant IS (importance sampling), annealé de beta_start → 1
    """

    def __init__(self, capacity=100_000, alpha=0.6, beta_start=0.4, beta_steps=200_000):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_steps  = beta_steps
        self._step       = 0

        self.tree_size   = 1
        while self.tree_size < capacity:
            self.tree_size *= 2

        self.sum_tree    = np.zeros(2 * self.tree_size, dtype=np.float64)
        self.min_tree    = np.full( 2 * self.tree_size, np.inf, dtype=np.float64)
        self.data        = [None] * capacity
        self.write_pos   = 0
        self.size        = 0
        self.max_prio    = 1.0

    @property
    def beta(self):
        t = min(self._step / self.beta_steps, 1.0)
        return self.beta_start + t * (1.0 - self.beta_start)

    def _update_tree(self, idx, prio):
        node = idx + self.tree_size
        self.sum_tree[node] = prio
        self.min_tree[node] = prio
        node //= 2
        while node >= 1:
            self.sum_tree[node] = self.sum_tree[2*node] + self.sum_tree[2*node+1]
            self.min_tree[node] = min(  self.min_tree[2*node],  self.min_tree[2*node+1])
            node //= 2

    def add(self, transition):
        """transition = (obs, action, reward, next_obs, done)"""
        prio = self.max_prio ** self.alpha
        self.data[self.write_pos] = transition
        self._update_tree(self.write_pos, prio)
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size      = min(self.size + 1, self.capacity)

    def _sample_idx(self, val):
        node = 1
        while node < self.tree_size:
            l = 2 * node
            if val <= self.sum_tree[l]:
                node = l
            else:
                val -= self.sum_tree[l]
                node = l + 1
        return node - self.tree_size

    def sample(self, batch_size):
        self._step += 1
        idxs    = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)
        total   = self.sum_tree[1]
        min_p   = self.min_tree[1] / total
        max_w   = (min_p * self.size) ** (-self.beta)

        segment = total / batch_size
        for i in range(batch_size):
            val       = np.random.uniform(segment * i, segment * (i + 1))
            idx       = self._sample_idx(val)
            idxs[i]   = idx
            p_i       = self.sum_tree[idx + self.tree_size] / total
            weights[i] = ((p_i * self.size) ** (-self.beta)) / max_w

        transitions = [self.data[i] for i in idxs]
        obs, actions, rewards, next_obs, dones = zip(*transitions)

        return (
            np.stack(obs).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_obs).astype(np.float32),
            np.array(dones,   dtype=np.float32),
            idxs,
            weights,
        )

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            prio = (abs(err) + 1e-6) ** self.alpha
            self.max_prio = max(self.max_prio, prio)
            self._update_tree(idx, prio)

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
#  RÉSEAU DUELING DQN  (torch)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_dqn(torch_nn):
    nn = torch_nn

    class DuelingDQN(nn.Module):
        """
        Dueling DQN : sépare l'estimation de la valeur d'état V(s)
        et de l'avantage A(s,a), puis les combine :
            Q(s,a) = V(s) + A(s,a) - mean(A(s,·))
        """
        def __init__(self, n_feat=N_FEATURES, n_actions=N_ACTIONS, hidden=256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 128),
                nn.ReLU(),
            )
            # Branche Value
            self.value_stream = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
            # Branche Advantage
            self.adv_stream = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_actions),
            )
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            shared = self.shared(x)
            value  = self.value_stream(shared)              # (B, 1)
            adv    = self.adv_stream(shared)                # (B, A)
            q      = value + adv - adv.mean(dim=-1, keepdim=True)
            return q

    return DuelingDQN()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT DQN
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    save_path          = "agent_dqn_weights.npz",
    total_steps        = 1_500_000,
    buffer_size        = 100_000,
    batch_size         = 256,
    lr                 = 1e-4,
    gamma              = 0.995,
    epsilon_start      = 1.0,
    epsilon_end        = 0.05,
    epsilon_decay_steps= 500_000,
    target_update_freq = 1_000,     # steps entre mises à jour du réseau cible
    tau                = 0.005,     # soft update coefficient
    warmup_steps       = 5_000,     # steps avant le premier apprentissage
    train_freq         = 4,         # apprentissage tous les N steps
    per_alpha          = 0.6,
    per_beta_start     = 0.4,
    per_beta_steps     = 600_000,
    log_interval       = 20,        # épisodes
    checkpoint_interval= 100,       # épisodes
    scenarios          = ('training_1', 'training_2', 'training_3'),
    device_str         = "auto",
    n_envs             = 4,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch requis pour l'entraînement.")

    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from src.env_sailing import SailingEnv
        from src.wind_scenarios import get_wind_scenario
    except ImportError:
        from env_sailing import SailingEnv
        from wind_scenarios import get_wind_scenario

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    print(f"[DQN] Device : {device}")

    # ── Réseaux ───────────────────────────────────────────────────────────
    import torch.nn as nn_mod
    online_net = _build_dqn(nn_mod).to(device)
    target_net = _build_dqn(nn_mod).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    # ── Replay buffer ─────────────────────────────────────────────────────
    replay = PrioritizedReplayBuffer(
        capacity    = buffer_size,
        alpha       = per_alpha,
        beta_start  = per_beta_start,
        beta_steps  = per_beta_steps,
    )

    # ── Environnements ────────────────────────────────────────────────────
    def make_env(sc, seed):
        params = get_wind_scenario(sc)
        env    = SailingEnv(**params)
        env.seed(seed)
        return env

    envs    = [make_env(scenarios[i % len(scenarios)], i * 1337) for i in range(n_envs)]
    raw_obs = []
    for env in envs:
        o, _ = env.reset()
        raw_obs.append(o)

    # ── Métriques ─────────────────────────────────────────────────────────
    metrics = {
        'success_rate':      [],
        'collision_rate':    [],
        'mean_score':        [],
        'mean_steps_success':[],
        'mean_shaped_reward':[],
        'td_loss':           [],
        'epsilon':           [],
        'mean_q_value':      [],
    }

    ep_rewards    = [0.0]  * n_envs
    ep_lengths    = [0]    * n_envs
    ep_successes  = [False]* n_envs
    ep_collisions = [False]* n_envs
    ep_shaped     = [0.0]  * n_envs

    recent_success   = deque(maxlen=100)
    recent_collision = deque(maxlen=100)
    recent_reward    = deque(maxlen=100)
    recent_length    = deque(maxlen=100)
    recent_shaped    = deque(maxlen=100)
    recent_td        = deque(maxlen=500)
    recent_q         = deque(maxlen=500)

    completed_episodes = 0
    global_step        = 0
    t_start            = time.time()

    print(f"[DQN] Début entraînement : {total_steps:,} steps")

    while global_step < total_steps:
        for env_idx, env in enumerate(envs):
            # ── Epsilon-greedy ────────────────────────────────────────────
            eps = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end)
                * global_step / epsilon_decay_steps
            )

            obs_raw = raw_obs[env_idx]
            feat    = extract_features(obs_raw)

            if np.random.random() < eps:
                action = np.random.randint(N_ACTIONS)
            else:
                with torch.no_grad():
                    ft     = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
                    q_vals = online_net(ft)
                    action = q_vals.argmax(dim=1).item()
                    recent_q.append(q_vals.max().item())

            # ── Step ──────────────────────────────────────────────────────
            next_obs_raw, env_reward, terminated, truncated, info = env.step(action)
            is_stuck = info.get('is_stuck', False)

            sr, sr_comps = shaped_reward(obs_raw, next_obs_raw, env_reward, is_stuck)

            done = terminated or truncated

            # Stockage dans le replay
            replay.add((
                feat,
                action,
                sr,
                extract_features(next_obs_raw),
                float(done),
            ))

            ep_rewards[env_idx]    += env_reward
            ep_lengths[env_idx]    += 1
            ep_shaped[env_idx]     += sr
            if is_stuck:
                ep_collisions[env_idx] = True
            if env_reward > 50:
                ep_successes[env_idx] = True

            global_step += 1

            if done:
                recent_success.append(float(ep_successes[env_idx]))
                recent_collision.append(float(ep_collisions[env_idx]))
                recent_reward.append(ep_rewards[env_idx])
                recent_shaped.append(ep_shaped[env_idx])
                if ep_successes[env_idx]:
                    recent_length.append(ep_lengths[env_idx])

                completed_episodes += 1

                if completed_episodes % log_interval == 0:
                    elapsed = time.time() - t_start
                    mean_td = float(np.mean(recent_td)) if recent_td else 0.0
                    mean_q  = float(np.mean(recent_q))  if recent_q  else 0.0
                    print(
                        f"[Ep {completed_episodes:5d} | Step {global_step:7d} | "
                        f"{elapsed/60:.1f}min] "
                        f"Succès={np.mean(recent_success)*100:.1f}% | "
                        f"Collision={np.mean(recent_collision)*100:.1f}% | "
                        f"Score={np.mean(recent_reward):.2f} | "
                        f"Steps(succ)={np.mean(recent_length) if recent_length else 0:.1f} | "
                        f"ShapedR={np.mean(recent_shaped):.2f} | "
                        f"TD={mean_td:.4f} | Q={mean_q:.3f} | eps={eps:.3f}"
                    )
                    metrics['success_rate'].append(float(np.mean(recent_success)))
                    metrics['collision_rate'].append(float(np.mean(recent_collision)))
                    metrics['mean_score'].append(float(np.mean(recent_reward)))
                    metrics['mean_shaped_reward'].append(float(np.mean(recent_shaped)))
                    metrics['epsilon'].append(float(eps))
                    metrics['mean_q_value'].append(float(mean_q))
                    if recent_length:
                        metrics['mean_steps_success'].append(float(np.mean(recent_length)))

                if completed_episodes % checkpoint_interval == 0:
                    _save_weights(online_net, save_path, metrics)
                    print(f"  ✓ Checkpoint sauvegardé → {save_path}")

                # Reset
                new_o, _ = env.reset()
                raw_obs[env_idx]       = new_o
                ep_rewards[env_idx]    = 0.0
                ep_lengths[env_idx]    = 0
                ep_successes[env_idx]  = False
                ep_collisions[env_idx] = False
                ep_shaped[env_idx]     = 0.0
            else:
                raw_obs[env_idx] = next_obs_raw

            # ── Apprentissage ─────────────────────────────────────────────
            if (len(replay) >= warmup_steps and global_step % train_freq == 0):
                obs_b, act_b, rew_b, next_obs_b, done_b, idxs, weights_b = \
                    replay.sample(batch_size)

                obs_t      = torch.tensor(obs_b,      dtype=torch.float32).to(device)
                act_t      = torch.tensor(act_b,      dtype=torch.long).to(device)
                rew_t      = torch.tensor(rew_b,      dtype=torch.float32).to(device)
                next_obs_t = torch.tensor(next_obs_b, dtype=torch.float32).to(device)
                done_t     = torch.tensor(done_b,     dtype=torch.float32).to(device)
                weights_t  = torch.tensor(weights_b,  dtype=torch.float32).to(device)

                # Double DQN : sélection d'action avec online_net,
                #              évaluation avec target_net
                with torch.no_grad():
                    next_actions = online_net(next_obs_t).argmax(dim=1)
                    next_q       = target_net(next_obs_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    targets      = rew_t + gamma * next_q * (1.0 - done_t)

                current_q = online_net(obs_t).gather(1, act_t.unsqueeze(1)).squeeze(1)
                td_errors = (current_q - targets).detach().cpu().numpy()

                # Perte Huber pondérée par IS
                loss = (weights_t * F.huber_loss(current_q, targets, reduction='none')).mean()

                optimizer.zero_grad()
                loss.backward()
                nn_mod.utils.clip_grad_norm_(online_net.parameters(), 10.0)
                optimizer.step()

                recent_td.append(loss.item())
                metrics['td_loss'].append(float(loss.item()))

                # Mise à jour priorités PER
                replay.update_priorities(idxs, td_errors)

                # Soft update du réseau cible
                if global_step % target_update_freq == 0:
                    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
                        p_target.data.copy_(tau * p_online.data + (1 - tau) * p_target.data)

    # ── Sauvegarde finale ─────────────────────────────────────────────────
    _save_weights(online_net, save_path, metrics)
    elapsed = time.time() - t_start
    print(f"\n[DQN] Entraînement terminé. Modèle sauvegardé → {save_path}")
    print(f"  Durée totale          : {elapsed/60:.1f} min")
    print(f"  Taux de succès final  : {np.mean(list(recent_success))*100:.1f}%")
    print(f"  Taux de collision fin : {np.mean(list(recent_collision))*100:.1f}%")
    return metrics


def _save_weights(model, path, metrics=None):
    """Sauvegarde les poids en .npz (numpy) pour inférence sans torch."""
    weights = {name: param.cpu().numpy() for name, param in model.state_dict().items()}
    if metrics:
        weights['_metrics_json'] = np.array([json.dumps(metrics)])
    np.savez(path, **weights)


# ═══════════════════════════════════════════════════════════════════════════════
#  INFÉRENCE NUMPY-ONLY  — Dueling DQN forward pass
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyDuelingDQN:
    """
    Forward pass du Dueling DQN avec numpy uniquement.
    Charge les poids depuis le dict produit par model.state_dict().
    """

    def __init__(self, weights: dict):
        # Réseau partagé : 3 couches Linear + ReLU
        self.shared_layers = [
            (weights['shared.0.weight'], weights['shared.0.bias']),
            (weights['shared.2.weight'], weights['shared.2.bias']),
            (weights['shared.4.weight'], weights['shared.4.bias']),
        ]
        # Branche value : 2 couches
        self.value_layers = [
            (weights['value_stream.0.weight'], weights['value_stream.0.bias']),
            (weights['value_stream.2.weight'], weights['value_stream.2.bias']),
        ]
        # Branche advantage : 2 couches
        self.adv_layers = [
            (weights['adv_stream.0.weight'], weights['adv_stream.0.bias']),
            (weights['adv_stream.2.weight'], weights['adv_stream.2.bias']),
        ]

    @staticmethod
    def _relu(x):
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (N_FEATURES,). Retourne Q-values (N_ACTIONS,)."""
        h = x.astype(np.float32)
        for i, (W, b) in enumerate(self.shared_layers):
            h = self._relu(W @ h + b)

        # Value stream
        v = h.copy()
        for i, (W, b) in enumerate(self.value_layers):
            v = W @ v + b
            if i < len(self.value_layers) - 1:
                v = self._relu(v)

        # Advantage stream
        a = h.copy()
        for i, (W, b) in enumerate(self.adv_layers):
            a = W @ a + b
            if i < len(self.adv_layers) - 1:
                a = self._relu(a)

        # Dueling combination : Q = V + A - mean(A)
        q = v + a - a.mean()
        return q

    def act_greedy(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSE AGENT (hérite de BaseAgent)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from agents.base_agent import BaseAgent as _BaseAgent
except ImportError:
    try:
        from base_agent import BaseAgent as _BaseAgent
    except ImportError:
        class _BaseAgent:
            def __init__(self): pass
            def reset(self): pass
            def seed(self, seed=None): pass


class MyAgent(_BaseAgent):
    """
    Agent DQN entraîné pour le Sailing Challenge.

    Utilise uniquement numpy pour l'inférence (compatible Codabench).
    Les poids sont chargés depuis un fichier .npz.

    Usage :
        agent = MyAgent()                          # charge agent_dqn_weights.npz si présent
        agent = MyAgent("mon_chemin/poids.npz")    # chemin explicite
        action = agent.act(observation)
    """

    DEFAULT_WEIGHTS_PATH = "agent_dqn_weights.npz"

    def __init__(self, weights_path: str = None):
        super().__init__()
        self.np_random = np.random.default_rng()
        self._net: NumpyDuelingDQN = None

        path = weights_path or self.DEFAULT_WEIGHTS_PATH
        if os.path.exists(path):
            self.load(path)

    def act(self, observation: np.ndarray) -> int:
        feat = extract_features(observation)
        if self._net is not None:
            return self._net.act_greedy(feat)
        return self._fallback_act(observation)

    def _fallback_act(self, obs: np.ndarray) -> int:
        """Heuristique de secours si les poids ne sont pas disponibles."""
        x, y = int(obs[0]), int(obs[1])
        xl, xr, yb, yt = ISLAND_RECT
        # Contournement de l'île si nécessaire
        if y < yb and x > xl - 5 and x < xr + 5:
            return 2 if x < GOAL_X else 6
        gx, gy = GOAL_X - x, GOAL_Y - y
        if abs(gx) < 2 and gy > 0: return 0
        if gx > 0  and  gy > 0:    return 1
        if gx > 0:                  return 2
        if gx < 0  and  gy > 0:    return 7
        if gx < 0:                  return 6
        return 0

    def reset(self) -> None:
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        if self._net is None:
            print("[MyAgent] Aucun réseau chargé, rien à sauvegarder.")
            return
        weights = {}
        for i, (W, b) in enumerate(self._net.shared_layers):
            weights[f'shared.{i*2}.weight'] = W
            weights[f'shared.{i*2}.bias']   = b
        for i, (W, b) in enumerate(self._net.value_layers):
            weights[f'value_stream.{i*2}.weight'] = W
            weights[f'value_stream.{i*2}.bias']   = b
        for i, (W, b) in enumerate(self._net.adv_layers):
            weights[f'adv_stream.{i*2}.weight'] = W
            weights[f'adv_stream.{i*2}.bias']   = b
        np.savez(path, **weights)
        print(f"[MyAgent] Poids sauvegardés → {path}")

    def load(self, path: str) -> None:
        try:
            data    = np.load(path, allow_pickle=True)
            weights = {k: data[k] for k in data.files if not k.startswith('_')}
            self._net = NumpyDuelingDQN(weights)
            print(f"[MyAgent] Poids chargés depuis {path}")
        except Exception as e:
            print(f"[MyAgent] Impossible de charger {path} : {e}")
            self._net = None


# ═══════════════════════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    weights_path = "agent_dqn_weights.npz",
    n_episodes   = 50,
    scenarios    = ('training_1', 'training_2', 'training_3'),
    seed_offset  = 9999,
):
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from src.env_sailing import SailingEnv
        from src.wind_scenarios import get_wind_scenario
    except ImportError:
        from env_sailing import SailingEnv
        from wind_scenarios import get_wind_scenario

    agent = MyAgent(weights_path=weights_path)

    results = {sc: {'success': 0, 'collision': 0, 'rewards': [], 'lengths': []}
               for sc in scenarios}
    total_success = total_collision = 0
    all_rewards   = []

    for ep in range(n_episodes):
        sc     = scenarios[ep % len(scenarios)]
        params = get_wind_scenario(sc)
        env    = SailingEnv(**params)
        env.seed(seed_offset + ep)

        obs, _ = env.reset()
        agent.reset()
        done       = False
        ep_reward  = 0.0
        discount   = 1.0
        steps      = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += discount * reward
            discount  *= 0.995
            steps     += 1
            done        = terminated or truncated

        success   = ep_reward > 50
        collision = info.get('is_stuck', False)
        results[sc]['rewards'].append(ep_reward)
        results[sc]['success']   += int(success)
        results[sc]['collision'] += int(collision)
        if success:
            results[sc]['lengths'].append(steps)
        total_success   += int(success)
        total_collision += int(collision)
        all_rewards.append(ep_reward)

    print("\n" + "═" * 60)
    print("RÉSULTATS D'ÉVALUATION — DQN")
    print("═" * 60)
    print(f"  Épisodes       : {n_episodes}")
    print(f"  Taux succès    : {total_success/n_episodes*100:.1f}%")
    print(f"  Taux collision : {total_collision/n_episodes*100:.1f}%")
    print(f"  Score moyen    : {np.mean(all_rewards):.3f}")
    all_lengths = [l for sc in scenarios for l in results[sc]['lengths']]
    if all_lengths:
        print(f"  Steps (succès) : {np.mean(all_lengths):.1f}")
    print()
    for sc in scenarios:
        r = results[sc]
        n = n_episodes // len(scenarios)
        if n == 0: continue
        print(f"  [{sc}]")
        print(f"    Succès    : {r['success']}/{n} ({r['success']/max(n,1)*100:.0f}%)")
        print(f"    Collision : {r['collision']}/{n}")
        print(f"    Score moy : {np.mean(r['rewards']):.3f}")
        if r['lengths']:
            print(f"    Steps(succ): {np.mean(r['lengths']):.1f}")
        print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent DQN Dueling+PER — Sailing Challenge")
    parser.add_argument("--train",      action="store_true")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--weights",    type=str,   default="agent_dqn_weights.npz")
    parser.add_argument("--steps",      type=int,   default=1_500_000)
    parser.add_argument("--n_envs",     type=int,   default=4)
    parser.add_argument("--n_eval",     type=int,   default=50)
    parser.add_argument("--device",     type=str,   default="auto")
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch",      type=int,   default=256)
    parser.add_argument("--buffer",     type=int,   default=100_000)
    parser.add_argument("--eps_steps",  type=int,   default=500_000,
                        help="Steps pour décroissance epsilon")
    parser.add_argument("--log_interval", type=int, default=20)
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(0)

    if args.train:
        print("=" * 60)
        print("ENTRAÎNEMENT DQN (Dueling + Double + PER)")
        print("=" * 60)
        metrics = train(
            save_path           = args.weights,
            total_steps         = args.steps,
            n_envs              = args.n_envs,
            lr                  = args.lr,
            batch_size          = args.batch,
            buffer_size         = args.buffer,
            epsilon_decay_steps = args.eps_steps,
            log_interval        = args.log_interval,
            device_str          = args.device,
        )
        metrics_path = args.weights.replace(".npz", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Métriques sauvegardées → {metrics_path}")

    if args.eval:
        print("=" * 60)
        print("ÉVALUATION — DQN")
        print("=" * 60)
        evaluate(weights_path=args.weights, n_episodes=args.n_eval)