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

# Directions possibles (N, NE, E, SE, S, SO, O, NO, Idle)
EXPERT_DIRS = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,0]], dtype=np.float32)

def get_expert_action(obs_raw):
    """Calcule l'action Windmaster à partir d'une observation brute."""
    px, py = float(obs_raw[0]), float(obs_raw[1])
    vx, vy = float(obs_raw[2]), float(obs_raw[3])
    
    # Extraction du vent et de la carte
    wf = obs_raw[6:6+128*128*2].reshape(128,128,2).astype(np.float32)
    world_map = obs_raw[6+128*128*2:].reshape(128,128).astype(np.float32)
    
    # Vent local
    x_idx, y_idx = max(0, min(127, int(round(px)))), max(0, min(127, int(round(py))))
    wx, wy = float(wf[y_idx, x_idx, 0]), float(wf[y_idx, x_idx, 1])
    
    tgx, tgy = 64.0 - px, 127.0 - py
    dist = np.sqrt(tgx**2 + tgy**2)
    if dist < 1e-6: return 8
    
    tgx /= dist; tgy /= dist
    is_final = dist < 5.0
    best_a, best_s = 8, -1e18
    
    for i in range(8):
        # On utilise ta fonction _sailing_efficiency déjà présente dans le script
        dx, dy = EXPERT_DIRS[i]
        # Simulation simplifiée du step pour l'expert
        # (Tu peux copier la fonction _step de baseline.py si tu veux une précision 100%)
        vmg = vx * tgx + vy * tgy 
        
        # Scoring simplifié pour l'expert (VMG + Evitement île)
        safety = 1.0
        # Check collision basique
        if world_map[min(127, max(0, int(py+dy*5))), min(127, max(0, int(px+dx*5)))] == 1:
            safety = -10.0
            
        score = vmg * safety
        if score > best_s:
            best_s = score
            best_a = i
            
    return best_a


def pretrain_bc(model, envs, device, n_epochs=5, steps_per_epoch=2000, lr=1e-3):
    print(f"[BC] Initialisation par imitation (Warm-up)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for _ in range(steps_per_epoch):
            # 1. Choisir un environnement et obtenir l'expert
            idx = np.random.randint(len(envs))
            env = envs[idx]
            
            # On récupère l'observation brute actuelle de l'env (stockée dans raw_obs dans train)
            # Pour faire simple ici, on reset ou on sample
            obs_raw, _ = env.reset() 
            
            # 2. Calculer l'action de l'expert
            expert_act = get_expert_action(obs_raw)
            
            # 3. Prédiction du modèle (via features)
            feat = torch.tensor(extract_features(obs_raw)).to(device).unsqueeze(0)
            logits, _ = model(feat)
            
            # 4. Optimisation
            loss = loss_fn(logits, torch.tensor([expert_act]).to(device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoque {epoch+1}/{n_epochs} | Loss moyenne: {total_loss/steps_per_epoch:.4f}")


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
    use_bc=True,
    bc_epochs=3,
    steps_per_epoch=5000

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

    if use_bc:
        pretrain_bc(model, envs, device, n_epochs=bc_epochs, steps_per_epoch=steps_per_epoch)

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

    print(f"[PPO] Début entraînement : {total_steps:,} steps, {n_updates} updates")
    t_start = time.time()

    for update in range(1, n_updates + 1):
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

    DEFAULT_WEIGHTS_PATH = "ppo_bc.npz"

    def __init__(self, weights_path: str = None):
        super().__init__()
        self.np_random = np.random.default_rng()
        self._net: NumpyActorCritic = None

        path = weights_path or self.DEFAULT_WEIGHTS_PATH
        import base64
        import zlib
        import io
        try:
            compressed_bytes = base64.b64decode(WEIGHTS_B64)
            npz_bytes = zlib.decompress(compressed_bytes)
            with io.BytesIO(npz_bytes) as f:
                data = np.load(f, allow_pickle=True)
                weights = {k: data[k] for k in data.files if not k.startswith('_')}
            self._net = NumpyActorCritic(weights)
            print("[MyAgent] Succès : Poids chargés depuis la mémoire (Base64) !")
        except Exception as e:
            print(f"[MyAgent] Erreur critique : {e}")
            self._net = None

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
        """Charge les poids directement depuis la chaîne de caractères Base64 embarquée."""
        import base64
        import zlib
        import io
        try:
            compressed_bytes = base64.b64decode(WEIGHTS_B64)
            npz_bytes = zlib.decompress(compressed_bytes)
            with io.BytesIO(npz_bytes) as f:
                data = np.load(f, allow_pickle=True)
                weights = {k: data[k] for k in data.files if not k.startswith('_')}
            self._net = NumpyActorCritic(weights)
            print("[MyAgent] Succès : Poids chargés depuis la mémoire (Base64) !")
        except Exception as e:
            print(f"[MyAgent] Erreur critique : {e}")
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
    parser.add_argument("--use_bc", type=int, default=True,
                        help="Use to enable BC")
    parser.add_argument("--bc_epochs", type=int, default=5,
                        help="n_epochs for BC")
    parser.add_argument("--steps_per_epoch", type=int, default=5000,
                        help="steps_per_epoch for BC")
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
            use_bc=args.use_bc,
            bc_epochs=args.bc_epochs,
            steps_per_epoch=args.steps_per_epoch
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

# --- POIDS EMBARQUÉS ---
WEIGHTS_B64 = """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D33YjaOnXHHX0VcvePvChR/8j9HnD71kfL/r77JwwKNOW7xlh31Hj/vXN83u/fTB+Is7PXZ00FsvGA+vvnDhQ3s+YHzucQ8aXfWwi0fvPP/Bo91P/tn+5/X/bvYr+35u4ZgHXzx69rqHzl37N/8yesbf7TR786UHzM4ctsPi7/9ox8WPX/z08ZFvePXospP2XfznM/eYO23+NePHHD9afMMNn5nZ6dWvmr3sGW8a7bjTEaPelxqzF7/6otG179xr9OM3/nD8haftMt74748a/+0uN87esqa/uN9z7rJww80fW3jN3OnjPf5x09yr7nXl+NJN14x//I6XLTznnFtG//aGTXN//LBNoyu+fsholx1ePPvuo24cH/alC8YHv38wvvHPvzLe/6VvGd/1/E+Nb3zITnMb//Wi8bO+e/eFh13++6O33WXD+PPnvmbhmpu2G+35+OeOXrb/2rlP/ttui2990I9GsztfP/7G47Zf/OrHvjF63otOnH3B3LtHL3hRf+F9rSPGl192t7l/Of294xvu/9jxdmf099+tffzoAX/w0Lkfffhzo8/+1rmjzSdcMfvLH+6w8OS1rxnvd+qDxl98//MXzzzv3ePP/eKQxRsecs/Fo9666+hpRy5evv7ChfGpe71+/LUfnT6+efHTo43feODiB+72y4X933v9fk9+xkMW/2KXr85e89wLxq87+89mX/hH9x+dcelwfPb2+4xe9c2fjq/+6WdGn/2bk0bf2eOZc+96xHnjp73nPosfP/Xhc5fOfWy87h92mHvFYd8an3X8e2f3ffkjRm8+bjT69C+Hc7/c+JC5v37LRaPv/MuG0cOP+8X41IUTPvFnOxw4PvfAl43+4KIXX37lDr+98KAPvX/2iBu+ubDnLefOHvGrN46P/c6HP7H2qp/NvvaCT44+8c3u6E8OOnb0qaf8ZHT4t386/vkx+88e/dRrRx/6/A5zT37k42f/z2MuWjj0wMbiT669YnzAp++1eM6Bh88euN0Zo8/Mvnf8hc1XjL5wn7XjM//hE+NNu7994Q+6X778t/7XF8f3/tEeoz/61EPmXrXmCbPn7nrOaPvR/x4/9DkXzb5jzVXj9z9+4/43Nt84u9cevzd37xv+evaTb//s+PJHf3z8t+f/cHzte35r7sWPuGX2r9b0xlcdtM/ih486euG6fV83OuBNVy1cteuz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"""
