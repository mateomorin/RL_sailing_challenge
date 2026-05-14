"""
PPO Agent with Manual Wind Features — Sailing Challenge
========================================================

Architecture :
  - Extraction de ~40 features compactes depuis l'observation brute
    (position normalisée, vitesse, vent local, vent moyen par zone,
     vent prédit au prochain pas, distances à l'île et au goal,
     détection automatique de la meilleure route N/S)
  - Réseau Actor-Critic MLP  [40 → 128 → 128] partagé + têtes séparées
  - PPO-clip avec GAE, entropy bonus, value clipping
  - Shaped reward intermédiaire (progression + pénalité île)
  - Inférence numpy-only pour la soumission Codabench

Utilisation :
  python agent_ppo.py --train           # entraîne et sauvegarde
  python agent_ppo.py --eval            # évalue le modèle sauvegardé
  python agent_ppo.py --train --eval    # les deux

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

# ─── Imports conditionnels torch (entraînement uniquement) ───────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION  (numpy, partagé entraînement + inférence)
# ═══════════════════════════════════════════════════════════════════════════════

GRID_SIZE   = 128
GOAL_X      = 64
GOAL_Y      = 127

# Géométrie de l'île (pentagonale : rectangle + triangle pointant vers le bas)
# Rectangle : x ∈ [45,83], y ∈ [45,83]
# Triangle  : sommet à (64, 22), base en y=45
ISLAND_RECT = (45, 83, 45, 83)   # x_min, x_max, y_min, y_max
ISLAND_TIP_Y = 22                 # pointe basse du triangle

def _wind_at(wind_field_2d, x, y):
    """Vent au point (x,y) avec clipping aux bords."""
    xi = int(np.clip(x, 0, GRID_SIZE - 1))
    yi = int(np.clip(y, 0, GRID_SIZE - 1))
    return wind_field_2d[yi, xi]   # shape (2,)

def _predict_next_wind(wind_field_2d, mean_rot_deg=3.0):
    """
    Prédit le champ de vent au prochain pas en appliquant la rotation moyenne.
    Identique à _update_wind_field de l'env (déterministe, sans bruit).
    Renvoie un champ (128,128,2).
    """
    theta = np.deg2rad(mean_rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = wind_field_2d[:, :, 0]
    y = wind_field_2d[:, :, 1]
    next_u = x * cos_t - y * sin_t
    next_v = x * sin_t + y * cos_t
    return np.stack([next_u, next_v], axis=-1)

def _zone_mean_wind(wind_field_2d, x_min, x_max, y_min, y_max):
    """Vent moyen sur une sous-zone rectangulaire."""
    x_min = int(np.clip(x_min, 0, GRID_SIZE - 1))
    x_max = int(np.clip(x_max, 0, GRID_SIZE - 1)) + 1
    y_min = int(np.clip(y_min, 0, GRID_SIZE - 1))
    y_max = int(np.clip(y_max, 0, GRID_SIZE - 1)) + 1
    zone = wind_field_2d[y_min:y_max, x_min:x_max]
    if zone.size == 0:
        return np.zeros(2)
    return zone.reshape(-1, 2).mean(axis=0)

def _sailing_efficiency(boat_dir, wind_dir):
    """Efficacité de voile (réplique de sailing_physics.py, numpy-only)."""
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
        eff = 1.0 - 0.5 * (angle - 3 * np.pi / 4) / (np.pi / 4)
        return max(0.5, eff)

def extract_features(obs: np.ndarray) -> np.ndarray:
    """
    Transforme l'observation brute (≈65K floats) en vecteur de ~46 features.

    Features :
      [0-1]  position normalisée (x/128, y/128)
      [2-3]  vitesse (vx, vy) — déjà petits entiers
      [4-5]  vent local normalisé
      [6]    efficacité de voile : vent local vs vitesse actuelle
      [7-8]  vecteur vers le goal normalisé
      [9]    distance euclidienne au goal / 128
      [10]   distance à la bordure nord de l'île (y_min_island - y)  / 128
      [11]   distance à la bordure sud de l'île (y - y_max_island)   / 128
      [12]   distance à la bordure gauche de l'île (x - x_min_island) / 128
      [13]   distance à la bordure droite de l'île (x_max_island - x) / 128
      [14]   indicateur route NORD (1 si y > y_max_island, 0 sinon)
      [15]   indicateur route SUD  (1 si y < y_min_island, 0 sinon)
      [16-17] vent moyen couloir NORD (y ∈ [84,127], x ∈ [0,127]) normalisé
      [18-19] vent moyen couloir SUD  (y ∈ [0,44],  x ∈ [0,127]) normalisé
      [20-21] vent moyen zone goal (x∈[44,84], y∈[84,127]) normalisé
      [22-23] vent prédit t+1 à position actuelle normalisé
      [24-29] efficacité de voile pour 6 directions clés (N,NE,E,SE,S,O)
              vis-à-vis du vent local actuel
      [30]   angle vent local (normalisé -1..1)
      [31]   magnitude vent local / 10
      [32]   step_progress (estimé via distance parcourue, non dispo → 0)
      [33-38] vent à 6 points d'anticipation sur la trajectoire directe
      [39-42] vent moyen 4 quadrants (NO, NE, SO, SE) pour détecter asymétrie
      [43]   avantage route NORD vs SUD (eff_nord - eff_sud)
      [44]   magnitude vitesse / max_speed
      [45]   indicateur "près de l'île" (1 si dist_min < 10 cells)
    """
    # ── Décodage de l'observation ──────────────────────────────────────────
    x, y     = float(obs[0]), float(obs[1])
    vx, vy   = float(obs[2]), float(obs[3])
    wx, wy   = float(obs[4]), float(obs[5])

    wind_flat  = obs[6:6 + GRID_SIZE * GRID_SIZE * 2]
    world_flat = obs[6 + GRID_SIZE * GRID_SIZE * 2:]

    wind_field = wind_flat.reshape(GRID_SIZE, GRID_SIZE, 2)
    # world_map  = world_flat.reshape(GRID_SIZE, GRID_SIZE)  # non utilisée ici

    wind_speed = np.sqrt(wx**2 + wy**2) + 1e-9

    # ── Features de base ──────────────────────────────────────────────────
    feat = []

    # [0-1] position normalisée
    feat.extend([x / GRID_SIZE, y / GRID_SIZE])

    # [2-3] vitesse
    feat.extend([vx, vy])

    # [4-5] vent local normalisé
    feat.extend([wx / wind_speed, wy / wind_speed])

    # [6] efficacité voile actuelle
    v_speed = np.sqrt(vx**2 + vy**2)
    if v_speed > 0.1:
        eff_cur = _sailing_efficiency(np.array([vx, vy]), np.array([wx, wy]))
    else:
        eff_cur = 0.0
    feat.append(eff_cur)

    # [7-9] direction et distance vers le goal
    gx, gy = GOAL_X - x, GOAL_Y - y
    dist_goal = np.sqrt(gx**2 + gy**2) + 1e-9
    feat.extend([gx / dist_goal, gy / dist_goal, dist_goal / GRID_SIZE])

    # [10-13] distances aux bords de l'île
    xl, xr, yb, yt = ISLAND_RECT   # x_min, x_max, y_min, y_max
    feat.extend([
        (yb - y) / GRID_SIZE,          # bord bas de l'île (positif = en dessous)
        (y - yt) / GRID_SIZE,          # bord haut de l'île (positif = au dessus)
        (x - xl) / GRID_SIZE,          # bord gauche
        (xr - x) / GRID_SIZE,          # bord droit
    ])

    # [14-17] indicateurs de route
    above_island = float(y > yt)
    below_island = float(y < yb)
    left_island = float(x < xl)
    right_island = float(x > xr)
    feat.extend([above_island, below_island, left_island, right_island])

    # [18-29] vent moyen des couloirs selon l'ile (above_west, above_east, below_west, below_east, left, right)
    wind_aw = _zone_mean_wind(wind_field, 0, 63, yt + 1, 127)
    wind_ae = _zone_mean_wind(wind_field, 64, 127, yt + 1, 127)
    wind_bw = _zone_mean_wind(wind_field, 0, 63, yb - 1, 127)
    wind_be = _zone_mean_wind(wind_field, 64, 127, yb - 1, 127)
    wind_l = _zone_mean_wind(wind_field, 0, xl-1, yb, yt)
    wind_r = _zone_mean_wind(wind_field, 64, xr+1, yb, yt)
    wind_aw_spd = np.linalg.norm(wind_aw) + 1e-9
    wind_ae_spd = np.linalg.norm(wind_ae) + 1e-9
    wind_bw_spd = np.linalg.norm(wind_bw) + 1e-9
    wind_be_spd = np.linalg.norm(wind_be) + 1e-9
    wind_l_spd = np.linalg.norm(wind_l) + 1e-9
    wind_r_spd = np.linalg.norm(wind_r) + 1e-9
    feat.extend([wind_aw[0] / wind_aw_spd, wind_aw[1] / wind_aw_spd])
    feat.extend([wind_ae[0] / wind_ae_spd, wind_ae[1] / wind_ae_spd])
    feat.extend([wind_bw[0] / wind_bw_spd, wind_bw[1] / wind_bw_spd])
    feat.extend([wind_be[0] / wind_be_spd, wind_be[1] / wind_be_spd])
    feat.extend([wind_l[0] / wind_l_spd, wind_l[1] / wind_l_spd])
    feat.extend([wind_r[0] / wind_r_spd, wind_r[1] / wind_r_spd])

    # [30-31] vent moyen zone goal
    w_goal = _zone_mean_wind(wind_field, 44, 84, yt + 1, 127)
    w_goal_spd = np.linalg.norm(w_goal) + 1e-9
    feat.extend([w_goal[0] / w_goal_spd, w_goal[1] / w_goal_spd])

    # [32-33] vent prédit t+1 à position actuelle
    wind_next = _predict_next_wind(wind_field)
    wn = _wind_at(wind_next, x, y)
    wn_spd = np.linalg.norm(wn) + 1e-9
    feat.extend([wn[0] / wn_spd, wn[1] / wn_spd])

    # [34-39] efficacités pour 6 directions clés vs vent local
    key_dirs = [
        (0.0,  1.0),   # N
        (1.0,  1.0),   # NE
        (1.0,  0.0),   # E
        (0.0, -1.0),   # S
        (-1.0, 0.0),   # W
        (-1.0, 1.0),   # NW
    ]
    for dx, dy in key_dirs:
        feat.append(_sailing_efficiency(np.array([dx, dy]), np.array([wx, wy])))

    # [39-40] angle et magnitude vent local
    wind_angle = np.arctan2(wy, wx) / np.pi   # -1..1
    feat.extend([wind_angle, wind_speed / 10.0])

    # [41-42] placeholder step progress
    feat.append(0.0)

    # [43-48] vent à 6 points d'anticipation vers le goal
    n_waypoints = 6
    for i in range(1, n_waypoints + 1):
        t = i / (n_waypoints + 1)
        px = x + t * gx
        py = y + t * gy
        wp = _wind_at(wind_field, px, py)
        wp_spd = np.linalg.norm(wp) + 1e-9
        # Efficacité de voile si on allait directement vers le goal
        eff_wp = _sailing_efficiency(np.array([gx, gy]), wp)
        feat.append(eff_wp)
    # On garde seulement les 6 efficacités (indices 33-38)

    # [49-52] vent moyen 4 quadrants pour détecter asymétrie
    mid = GRID_SIZE // 2
    for (x0, x1, y0, y1) in [
        (0, mid, mid, GRID_SIZE - 1),    # NO
        (mid, GRID_SIZE - 1, mid, GRID_SIZE - 1),  # NE
        (0, mid, 0, mid),                # SO
        (mid, GRID_SIZE - 1, 0, mid),    # SE
    ]:
        wq = _zone_mean_wind(wind_field, x0, x1, y0, y1)
        feat.append(np.arctan2(wq[1], wq[0]) / np.pi)

    # [53] avantage route EST vs OUEST
    # Efficacité si on va plein nord depuis position actuelle dans chaque couloir
    eff_left_route = _sailing_efficiency(np.array([0.0, 1.0]), wind_l)
    eff_right_route = _sailing_efficiency(np.array([0.0, -1.0]), wind_r)
    feat.append(eff_left_route - eff_right_route)

    # [54] magnitude vitesse normalisée
    feat.append(v_speed / 8.0)

    # [55] near-island flag
    dist_to_island = min(
        abs(x - xl), abs(x - xr), abs(y - yb), abs(y - yt)
    )
    feat.append(float(dist_to_island < 10))

    return np.array(feat, dtype=np.float32)


N_FEATURES = 56


# ═══════════════════════════════════════════════════════════════════════════════
#  RÉSEAU ACTOR-CRITIC  (torch, entraînement uniquement)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_network(torch_module):
    """Construit le réseau Actor-Critic partagé."""
    nn   = torch_module
    class ActorCritic(nn.Module):
        def __init__(self, n_feat=N_FEATURES, n_actions=9, hidden=128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.actor_head  = nn.Linear(hidden, n_actions)
            self.critic_head = nn.Linear(hidden, 1)

            # Orthogonal init
            for layer in self.shared:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
            nn.init.zeros_(self.actor_head.bias)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.zeros_(self.critic_head.bias)

        def forward(self, x):
            shared = self.shared(x)
            logits = self.actor_head(shared)
            value  = self.critic_head(shared).squeeze(-1)
            return logits, value

        def get_action_and_value(self, x, action=None):
            logits, value = self(x)
            dist = Categorical(logits=logits)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()
            return action, log_prob, entropy, value

    return ActorCritic()


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAPED REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def shaped_reward(obs_prev, obs_curr, env_reward, terminated, is_stuck,
                  step, max_steps=500, gamma=0.995):
    """
    Reward façonné :
      - env_reward     : 100 si goal atteint
      - progression    : réduction de distance au goal (normalisée)
      - pénalité île   : si collision
      - pénalité temps : légère pénalité pour décourager l'agent de traîner
      - bonus efficacité : encourage l'exploitation du vent
    """
    x_p, y_p = obs_prev[0], obs_prev[1]
    x_c, y_c = obs_curr[0],  obs_curr[1]
    wx, wy   = obs_curr[4],  obs_curr[5]

    # Distance avant/après
    d_prev = np.sqrt((GOAL_X - x_p)**2 + (GOAL_Y - y_p)**2)
    d_curr = np.sqrt((GOAL_X - x_c)**2 + (GOAL_Y - y_c)**2)
    progress = (d_prev - d_curr) / GRID_SIZE * 5.0   # scale

    # Pénalité collision
    collision_penalty = -50.0 if is_stuck else 0.0

    # Légère pénalité temporelle
    time_penalty = -0.02

    # Bonus efficacité de voile (encourage les manœuvres utiles)
    vx, vy = obs_curr[2], obs_curr[3]
    v_spd = np.sqrt(vx**2 + vy**2)
    eff_bonus = 0.0
    if v_spd > 0.1:
        eff = _sailing_efficiency(np.array([vx, vy]), np.array([wx, wy]))
        eff_bonus = eff * 0.05

    total = env_reward + progress + collision_penalty + time_penalty + eff_bonus
    return total, {
        'progress': progress,
        'collision_penalty': collision_penalty,
        'time_penalty': time_penalty,
        'eff_bonus': eff_bonus,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT PPO
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    save_path="agent_ppo_bis_weights.npz",
    n_envs=8,
    total_steps=3_000_000,
    rollout_steps=512,
    n_epochs=4,
    minibatch_size=256,
    lr=3e-4,
    gamma=0.995,
    gae_lambda=0.95,
    clip_coef=0.2,
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    log_interval=20,          # épisodes
    checkpoint_interval=50,   # épisodes (sauvegarde)
    scenarios=('training_1', 'training_2', 'training_3'),
    device_str="auto",
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch requis pour l'entraînement.")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical

    # Imports locaux env
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from src.env_sailing import SailingEnv
        from src.wind_scenarios import get_wind_scenario
    except ImportError:
        # Fallback si le module est à la racine
        from env_sailing import SailingEnv
        from wind_scenarios import get_wind_scenario

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    print(f"[PPO] Device: {device}")

    # ── Création des environnements ────────────────────────────────────────
    def make_env(scenario_name, seed):
        params = get_wind_scenario(scenario_name)
        env = SailingEnv(**params)
        env.seed(seed)
        return env

    # n_envs environnements, répartis sur les scénarios
    envs = []
    for i in range(n_envs):
        sc = scenarios[i % len(scenarios)]
        envs.append(make_env(sc, seed=i * 1000))

    # ── Réseau ────────────────────────────────────────────────────────────
    model = _make_network(nn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # ── Buffers ───────────────────────────────────────────────────────────
    buf_obs     = torch.zeros(rollout_steps, n_envs, N_FEATURES).to(device)
    buf_actions = torch.zeros(rollout_steps, n_envs, dtype=torch.long).to(device)
    buf_logprobs= torch.zeros(rollout_steps, n_envs).to(device)
    buf_rewards = torch.zeros(rollout_steps, n_envs).to(device)
    buf_dones   = torch.zeros(rollout_steps, n_envs).to(device)
    buf_values  = torch.zeros(rollout_steps, n_envs).to(device)

    # ── État initial ──────────────────────────────────────────────────────
    obs_list = []
    for env in envs:
        o, _ = env.reset()
        obs_list.append(extract_features(o))
    next_obs   = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
    next_dones = torch.zeros(n_envs).to(device)

    # Raw obs pour le shaped reward
    raw_obs = [env.reset()[0] for env in envs]
    # Re-réinitialiser proprement
    raw_obs = []
    for env in envs:
        o, _ = env.reset()
        raw_obs.append(o)
    next_obs = torch.tensor(
        np.stack([extract_features(o) for o in raw_obs]), dtype=torch.float32
    ).to(device)

    # ── Métriques ─────────────────────────────────────────────────────────
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_rate': [],
        'collision_rate': [],
        'mean_shaped_reward': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': [],
        'approx_kl': [],
    }

    ep_rewards   = [0.0] * n_envs
    ep_lengths   = [0]   * n_envs
    ep_successes = [False] * n_envs
    ep_collisions= [False] * n_envs
    ep_shaped    = [0.0] * n_envs

    completed_episodes = 0
    recent_success  = deque(maxlen=100)
    recent_collision= deque(maxlen=100)
    recent_reward   = deque(maxlen=100)
    recent_length   = deque(maxlen=100)
    recent_shaped   = deque(maxlen=100)

    global_step = 0
    n_updates   = total_steps // (n_envs * rollout_steps)

    one_third_updates = n_updates // 3

    print(f"[PPO] Début entraînement : {total_steps:,} steps, {n_updates} updates")
    t_start = time.time()

    for update in range(1, n_updates + 1):
        # --- AJOUT : Gestion dynamique des scénarios ---
        if update == 1:
            # Phase 1 : Uniquement training_3
            current_scenarios = ['training_3']
            print(f"\n[Phase 1] Entraînement sur {current_scenarios} uniquement...")
            for i in range(n_envs):
                envs[i] = make_env('training_3', seed=i * 1000)
        
        elif update == one_third_updates + 1:
            # Phase 2 : Tous les scénarios
            current_scenarios = scenarios           # ('training_1', 'training_2', 'training_3')
            print(f"\n[Phase 2] Entraînement sur tous les scénarios : {current_scenarios}")
            for i in range(n_envs):
                sc = current_scenarios[i % len(current_scenarios)]
                envs[i] = make_env(sc, seed=i * 1000 + 999) # Nouveau seed pour la diversité
            
            # Reset de l'état pour les nouveaux envs
            obs_list = [extract_features(env.reset()[0]) for env in envs]
            next_obs = torch.tensor(np.stack(obs_list), dtype=torch.float32).to(device)
            raw_obs = [env.reset()[0] for env in envs] # Important pour le shaped_reward
        
        # Anneal lr linéaire
        frac = 1.0 - (update - 1) / n_updates
        lr_now = lr * frac
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # ── Collecte du rollout ───────────────────────────────────────────
        for step in range(rollout_steps):
            global_step += n_envs
            buf_obs[step]   = next_obs
            buf_dones[step] = next_dones

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(next_obs)
            buf_actions[step]  = action
            buf_logprobs[step] = log_prob
            buf_values[step]   = value

            # Step dans chaque env
            new_obs_list = []
            step_rewards  = []
            step_dones    = []
            for i, env in enumerate(envs):
                a = action[i].item()
                o_next, r, terminated, truncated, info = env.step(a)

                # Shaped reward
                sr, sr_components = shaped_reward(
                    raw_obs[i], o_next, r, terminated,
                    info.get('is_stuck', False), ep_lengths[i]
                )

                ep_rewards[i]    += r
                ep_lengths[i]    += 1
                ep_shaped[i]     += sr
                if info.get('is_stuck', False):
                    ep_collisions[i] = True
                if r > 50:  # goal reached
                    ep_successes[i] = True

                done = terminated or truncated
                step_rewards.append(sr)
                step_dones.append(float(done))
                raw_obs[i] = o_next

                if done:
                    recent_success.append(float(ep_successes[i]))
                    recent_collision.append(float(ep_collisions[i]))
                    recent_reward.append(ep_rewards[i])
                    recent_shaped.append(ep_shaped[i])
                    if ep_successes[i]:
                        recent_length.append(ep_lengths[i])

                    completed_episodes += 1

                    # Log
                    if completed_episodes % log_interval == 0:
                        elapsed = time.time() - t_start
                        print(
                            f"[Ep {completed_episodes:5d} | Step {global_step:7d} | "
                            f"{elapsed/60:.1f}min] "
                            f"Succès={np.mean(recent_success)*100:.1f}% | "
                            f"Collision={np.mean(recent_collision)*100:.1f}% | "
                            f"Reward={np.mean(recent_reward):.2f} | "
                            f"Steps(succ)={np.mean(recent_length) if recent_length else 0:.1f} | "
                            f"ShapedR={np.mean(recent_shaped):.2f} | "
                            f"lr={lr_now:.2e}"
                        )
                        metrics['success_rate'].append(float(np.mean(recent_success)))
                        metrics['collision_rate'].append(float(np.mean(recent_collision)))
                        metrics['mean_shaped_reward'].append(float(np.mean(recent_shaped)))

                    # Checkpoint
                    if completed_episodes % checkpoint_interval == 0:
                        _save_weights(model, save_path, metrics)
                        print(f"  ✓ Checkpoint sauvegardé → {save_path}")

                    # Reset env
                    new_o, _ = env.reset()
                    raw_obs[i] = new_o
                    ep_rewards[i]    = 0.0
                    ep_lengths[i]    = 0
                    ep_successes[i]  = False
                    ep_collisions[i] = False
                    ep_shaped[i]     = 0.0
                    o_next = new_o

                new_obs_list.append(extract_features(o_next))

            buf_rewards[step] = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            next_obs   = torch.tensor(np.stack(new_obs_list), dtype=torch.float32).to(device)
            next_dones = torch.tensor(step_dones, dtype=torch.float32).to(device)

        # ── Calcul des avantages GAE ───────────────────────────────────────
        with torch.no_grad():
            next_value = model.get_action_and_value(next_obs)[3]
            advantages = torch.zeros_like(buf_rewards)
            last_gae   = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    next_non_terminal = 1.0 - next_dones
                    nv = next_value
                else:
                    next_non_terminal = 1.0 - buf_dones[t + 1]
                    nv = buf_values[t + 1]
                delta     = buf_rewards[t] + gamma * nv * next_non_terminal - buf_values[t]
                last_gae  = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae
            returns = advantages + buf_values

        # ── Mise à jour PPO ────────────────────────────────────────────────
        b_obs      = buf_obs.reshape(-1, N_FEATURES)
        b_actions  = buf_actions.reshape(-1)
        b_logprobs = buf_logprobs.reshape(-1)
        b_advs     = advantages.reshape(-1)
        b_returns  = returns.reshape(-1)

        # Normalisation des avantages
        b_advs = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        n_samples  = rollout_steps * n_envs
        inds       = np.arange(n_samples)
        update_pg_losses, update_vf_losses, update_ents, update_kls = [], [], [], []

        for epoch in range(n_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_samples, minibatch_size):
                end  = start + minibatch_size
                mb   = inds[start:end]

                _, new_lp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )

                log_ratio   = new_lp - b_logprobs[mb]
                ratio       = log_ratio.exp()
                approx_kl   = ((ratio - 1) - log_ratio).mean().item()

                mb_adv = b_advs[mb]

                # Policy loss (PPO-clip)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_loss_unclipped = (new_val - b_returns[mb]).pow(2)
                v_loss  = 0.5 * v_loss_unclipped.mean()

                # Entropy bonus
                ent_loss = entropy.mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                update_pg_losses.append(pg_loss.item())
                update_vf_losses.append(v_loss.item())
                update_ents.append(ent_loss.item())
                update_kls.append(approx_kl)

        metrics['policy_loss'].append(float(np.mean(update_pg_losses)))
        metrics['value_loss'].append(float(np.mean(update_vf_losses)))
        metrics['entropy'].append(float(np.mean(update_ents)))
        metrics['approx_kl'].append(float(np.mean(update_kls)))

    # ── Sauvegarde finale ─────────────────────────────────────────────────
    _save_weights(model, save_path, metrics)
    print(f"\n[PPO] Entraînement terminé. Modèle sauvegardé → {save_path}")
    print(f"  Durée totale : {(time.time() - t_start)/60:.1f} min")
    print(f"  Taux de succès final : {np.mean(list(recent_success))*100:.1f}%")
    print(f"  Taux de collision final : {np.mean(list(recent_collision))*100:.1f}%")

    return metrics


def _save_weights(model, path, metrics=None):
    """Sauvegarde les poids du modèle en numpy (.npz) pour inférence sans torch."""
    weights = {}
    for name, param in model.state_dict().items():
        weights[name] = param.cpu().numpy()
    if metrics:
        weights['_metrics_json'] = np.array([json.dumps(metrics)])
    with fs.open("mamorin/rl_sailing/models/" + path, 'wb') as f:
        np.savez(f, **weights)


# ═══════════════════════════════════════════════════════════════════════════════
#  INFÉRENCE NUMPY-ONLY
#  Réimplémentation forward pass avec numpy uniquement
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyActorCritic:
    """
    Forward pass du réseau Actor-Critic avec numpy uniquement.
    Compatible avec les poids sauvegardés par _save_weights().
    """

    def __init__(self, weights: dict):
        # Couches partagées
        self.W0 = weights['shared.0.weight']   # (128, 46)
        self.b0 = weights['shared.0.bias']     # (128,)
        self.W2 = weights['shared.2.weight']   # (128, 128)
        self.b2 = weights['shared.2.bias']     # (128,)
        # Tête acteur
        self.Wa = weights['actor_head.weight'] # (9, 128)
        self.ba = weights['actor_head.bias']   # (9,)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _softmax(x):
        ex = np.exp(x - x.max())
        return ex / ex.sum()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (N_FEATURES,) float32
        Returns : probs (9,)
        """
        h = self._tanh(self.W0 @ x + self.b0)
        h = self._tanh(self.W2 @ h + self.b2)
        logits = self.Wa @ h + self.ba
        return self._softmax(logits)

    def act_greedy(self, x: np.ndarray) -> int:
        probs = self.forward(x)
        return int(np.argmax(probs))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSE AGENT (hérite de BaseAgent)
# ═══════════════════════════════════════════════════════════════════════════════

# Import conditionnel BaseAgent
try:
    from agents.base_agent import BaseAgent
except ImportError:
    try:
        from base_agent import BaseAgent
    except ImportError:
        from src.agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    """
    Agent PPO entraîné pour le Sailing Challenge.

    Utilise uniquement numpy pour l'inférence.
    Charge les poids depuis un fichier .npz.

    Usage :
        agent = MyAgent()
        agent.load("agent_ppo_bis_weights.npz")
        action = agent.act(observation)
    """

    DEFAULT_WEIGHTS_PATH = "agent_ppo_bis_weights.npz"

    def __init__(self, weights_path: str = None):
        super().__init__()
        self.np_random = np.random.default_rng()
        self._net: NumpyActorCritic = None

        path = weights_path or self.DEFAULT_WEIGHTS_PATH
        if os.path.exists(path):
            self.load(path)

    def act(self, observation: np.ndarray) -> int:
        """
        Sélectionne une action depuis l'observation brute.

        Si le modèle n'est pas chargé, utilise une heuristique de secours.
        """
        feat = extract_features(observation)

        if self._net is not None:
            return self._net.act_greedy(feat)
        else:
            # Heuristique de secours : aller vers le goal en évitant l'île
            return self._fallback_act(observation)

    def _fallback_act(self, obs: np.ndarray) -> int:
        """Heuristique simple si les poids ne sont pas disponibles."""
        x, y = int(obs[0]), int(obs[1])
        gx, gy = GOAL_X - x, GOAL_Y - y

        # Si en dessous de l'île, aller vers le nord en contournant
        xl, xr, yb, yt = ISLAND_RECT
        if y < yb and x > xl - 5 and x < xr + 5:
            # Longer le bord gauche ou droit selon la position
            if x < GOAL_X:
                return 2  # East pour dépasser l'île
            else:
                return 6  # West

        # Direction vers le goal
        if abs(gx) < 2 and gy > 0:
            return 0  # North
        if gx > 0 and gy > 0:
            return 1  # NE
        if gx > 0:
            return 2  # E
        if gx < 0 and gy > 0:
            return 7  # NW
        if gx < 0:
            return 6  # W
        return 0

    def reset(self) -> None:
        """Reset l'état interne de l'agent entre les épisodes."""
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)

    def save(self, path: str) -> None:
        """
        Sauvegarde les poids du modèle.
        (Appeler _save_weights séparément depuis torch lors de l'entraînement)
        """
        if self._net is not None:
            weights = {
                'shared.0.weight': self._net.W0,
                'shared.0.bias':   self._net.b0,
                'shared.2.weight': self._net.W2,
                'shared.2.bias':   self._net.b2,
                'actor_head.weight': self._net.Wa,
                'actor_head.bias':   self._net.ba,
            }
            np.savez(path, **weights)
            print(f"[MyAgent] Poids sauvegardés → {path}")

    def load(self, path: str) -> None:
        """Charge les poids depuis un fichier .npz."""
        try:
            data = np.load(path, allow_pickle=True)
            weights = {k: data[k] for k in data.files if not k.startswith('_')}
            self._net = NumpyActorCritic(weights)
            print(f"[MyAgent] Poids chargés depuis {path}")
        except Exception as e:
            print(f"[MyAgent] Impossible de charger {path}: {e}")
            self._net = None


# ═══════════════════════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(
    weights_path="agent_ppo_bis_weights.npz",
    n_episodes=50,
    scenarios=('training_1', 'training_2', 'training_3'),
    seed_offset=9999,
):
    """Évalue l'agent entraîné et affiche les métriques."""
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

    total_success = 0
    total_collision = 0
    all_rewards = []

    for ep in range(n_episodes):
        sc = scenarios[ep % len(scenarios)]
        params = get_wind_scenario(sc)
        env = SailingEnv(**params)
        env.seed(seed_offset + ep)

        obs, _ = env.reset()
        agent.reset()
        done = False
        ep_reward = 0.0
        discount = 1.0
        steps = 0

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += discount * reward
            discount  *= 0.995
            steps     += 1
            done = terminated or truncated

        success   = ep_reward > 50
        collision = info.get('is_stuck', False)

        results[sc]['success']   += int(success)
        results[sc]['collision'] += int(collision)
        results[sc]['rewards'].append(ep_reward)
        if success:
            results[sc]['lengths'].append(steps)

        total_success   += int(success)
        total_collision += int(collision)
        all_rewards.append(ep_reward)

    print("\n" + "═" * 60)
    print("RÉSULTATS D'ÉVALUATION")
    print("═" * 60)
    print(f"  Épisodes      : {n_episodes}")
    print(f"  Taux succès   : {total_success/n_episodes*100:.1f}%")
    print(f"  Taux collision: {total_collision/n_episodes*100:.1f}%")
    print(f"  Score moyen   : {np.mean(all_rewards):.3f}")
    if any(results[sc]['lengths'] for sc in scenarios):
        all_lengths = [l for sc in scenarios for l in results[sc]['lengths']]
        print(f"  Steps moyens (succès) : {np.mean(all_lengths):.1f}")
    print()

    for sc in scenarios:
        r = results[sc]
        n = n_episodes // len(scenarios)
        if n == 0:
            continue
        print(f"  [{sc}]")
        print(f"    Succès   : {r['success']}/{n} ({r['success']/n*100:.0f}%)")
        print(f"    Collision: {r['collision']}/{n}")
        print(f"    Score moy: {np.mean(r['rewards']):.3f}")
        if r['lengths']:
            print(f"    Steps(succ): {np.mean(r['lengths']):.1f}")
        print()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import s3fs
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )
    parser = argparse.ArgumentParser(description="Agent PPO — Sailing Challenge")
    parser.add_argument("--train",      action="store_true", help="Entraîner le modèle")
    parser.add_argument("--eval",       action="store_true", help="Évaluer le modèle")
    parser.add_argument("--weights",    type=str, default="agent_ppo_bis_weights.npz",
                        help="Chemin du fichier de poids (.npz)")
    parser.add_argument("--steps",      type=int, default=3_000_000,
                        help="Nombre total de steps d'entraînement")
    parser.add_argument("--n_envs",     type=int, default=20,
                        help="Nombre d'environnements parallèles")
    parser.add_argument("--n_eval",     type=int, default=50,
                        help="Nombre d'épisodes d'évaluation")
    parser.add_argument("--device",     type=str, default="auto",
                        help="Device torch : auto, cpu, cuda")
    parser.add_argument("--rollout",    type=int, default=512,
                        help="Steps par rollout")
    parser.add_argument("--lr",         type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--ent_coef",   type=float, default=0.01,
                        help="Coefficient d'entropie")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log toutes les N épisodes")
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(0)

    if args.train:
        print("=" * 60)
        print("ENTRAÎNEMENT PPO — Sailing Challenge")
        print("=" * 60)
        metrics = train(
            save_path=args.weights,
            total_steps=args.steps,
            n_envs=args.n_envs,
            rollout_steps=args.rollout,
            lr=args.lr,
            ent_coef=args.ent_coef,
            log_interval=args.log_interval,
            device_str=args.device,
        )

        # Sauvegarde des métriques JSON séparément
        metrics_path = args.weights.replace(".npz", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Métriques sauvegardées → {metrics_path}")

    if args.eval:
        print("=" * 60)
        print("ÉVALUATION — Sailing Challenge")
        print("=" * 60)
        evaluate(weights_path=args.weights, n_episodes=args.n_eval)