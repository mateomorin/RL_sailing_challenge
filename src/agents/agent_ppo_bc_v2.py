"""
PPO + BC Interleaved — Expert A* Wind-Aware + Curriculum + Random Scenarios
============================================================================

Améliorations vs agent_ppo_bc.py original :

1. EXPERT A* WIND-AWARE
   Dijkstra sur la grille complète 128×128 avec coût = 1 / efficacité_voile(cellule).
   L'expert choisit ainsi le chemin qui maximise réellement la vitesse selon le vent,
   contourne l'île automatiquement, et fonctionne sur N'IMPORTE quel scénario aléatoire.

2. BC INTERLEAVED (pas seulement en warm-up)
   À chaque update PPO, un mini-batch BC est intercalé avec un coefficient λ_bc
   qui décroît exponentiellement (bc_coef_start → bc_coef_end sur bc_decay_steps).
   Le modèle reste ancré sur de bonnes trajectoires tout en explorant.

3. CURRICULUM DE SCÉNARIOS
   Phase 1 (0 → curriculum_end_frac) : scénarios "faciles" — vent fort et favorable
     (amplitude > 0.7, direction principalement vers le goal)
   Phase 2 (curriculum_end_frac → fin) : tous les scénarios aléatoires sans restriction
   Évite que le modèle soit bloqué sur des épisodes nuls en début d'entraînement.

4. RANDOM SCENARIOS DÈS LE DÉBUT
   Chaque reset d'env tire un nouveau scénario aléatoire (avec cohérence spatiale).
   orig_ratio (défaut 0.15) garde un minimum de scénarios originaux.

Utilisation :
  python agent_ppo_bc_v2.py --train
  python agent_ppo_bc_v2.py --train --eval
  python agent_ppo_bc_v2.py --eval --weights agent_ppo_bc_v2_weights.npz

Dépendances entraînement : torch, numpy
Dépendances inférence    : numpy uniquement
"""

import argparse
import heapq
import json
import os
import sys
import time
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES & FEATURE EXTRACTION  (identiques à agent_ppo_bc.py)
# ═══════════════════════════════════════════════════════════════════════════════

GRID_SIZE    = 128
GOAL_X       = 64
GOAL_Y       = 127
ISLAND_RECT  = (45, 83, 45, 83)   # x_min, x_max, y_min, y_max
ISLAND_TIP_Y = 22

ACTION_DIRS = np.array([
    [0, 1], [1, 1], [1, 0], [1, -1],
    [0,-1], [-1,-1],[-1, 0],[-1, 1],
    [0, 0],
], dtype=np.float32)


def _wind_at(wind_field_2d, x, y):
    xi = int(np.clip(x, 0, GRID_SIZE - 1))
    yi = int(np.clip(y, 0, GRID_SIZE - 1))
    return wind_field_2d[yi, xi]

def _predict_next_wind(wind_field_2d, mean_rot_deg=3.0):
    theta = np.deg2rad(mean_rot_deg)
    c, s  = np.cos(theta), np.sin(theta)
    u, v  = wind_field_2d[:,:,0], wind_field_2d[:,:,1]
    return np.stack([u*c - v*s, u*s + v*c], axis=-1)

def _zone_mean_wind(wind_field_2d, x_min, x_max, y_min, y_max):
    x0 = int(np.clip(x_min, 0, GRID_SIZE-1))
    x1 = int(np.clip(x_max, 0, GRID_SIZE-1)) + 1
    y0 = int(np.clip(y_min, 0, GRID_SIZE-1))
    y1 = int(np.clip(y_max, 0, GRID_SIZE-1)) + 1
    zone = wind_field_2d[y0:y1, x0:x1]
    return zone.reshape(-1, 2).mean(axis=0) if zone.size else np.zeros(2)

def _sailing_efficiency(boat_dir, wind_dir):
    wn = np.linalg.norm(wind_dir)
    bn = np.linalg.norm(boat_dir)
    if wn < 1e-9 or bn < 1e-9:
        return 0.05
    cos_a = np.clip(np.dot(-wind_dir/wn, boat_dir/bn), -1.0, 1.0)
    angle = np.arccos(cos_a)
    if   angle < np.pi/4:       return 0.05
    elif angle < np.pi/2:       return 0.5 + 0.5*(angle - np.pi/4)/(np.pi/4)
    elif angle < 3*np.pi/4:     return 1.0
    else:                        return max(0.5, 1.0 - 0.5*(angle - 3*np.pi/4)/(np.pi/4))

def extract_features(obs: np.ndarray) -> np.ndarray:
    x,  y  = float(obs[0]), float(obs[1])
    vx, vy = float(obs[2]), float(obs[3])
    wx, wy = float(obs[4]), float(obs[5])

    wind_field = obs[6:6+GRID_SIZE*GRID_SIZE*2].reshape(GRID_SIZE, GRID_SIZE, 2)
    wind_speed = np.sqrt(wx**2 + wy**2) + 1e-9
    feat = []

    feat.extend([x/GRID_SIZE, y/GRID_SIZE, vx, vy])
    feat.extend([wx/wind_speed, wy/wind_speed])

    v_speed = np.sqrt(vx**2 + vy**2)
    feat.append(_sailing_efficiency(np.array([vx,vy]), np.array([wx,wy])) if v_speed > 0.1 else 0.0)

    gx, gy    = GOAL_X-x, GOAL_Y-y
    dist_goal = np.sqrt(gx**2 + gy**2) + 1e-9
    feat.extend([gx/dist_goal, gy/dist_goal, dist_goal/GRID_SIZE])

    xl, xr, yb, yt = ISLAND_RECT
    feat.extend([(yb-y)/GRID_SIZE, (y-yt)/GRID_SIZE, (x-xl)/GRID_SIZE, (xr-x)/GRID_SIZE])
    feat.extend([float(y>yt), float(y<yb), float(x<xl), float(x>xr)])

    for zone in [
        (0,63,yt+1,127), (64,127,yt+1,127),
        (0,63,0,yb-1),   (64,127,0,yb-1),
        (0,xl-1,yb,yt),  (64,xr+1,yb,yt),
    ]:
        w = _zone_mean_wind(wind_field, *zone)
        s = np.linalg.norm(w) + 1e-9
        feat.extend([w[0]/s, w[1]/s])

    wg  = _zone_mean_wind(wind_field, 44, 84, yt+1, 127)
    wgs = np.linalg.norm(wg) + 1e-9
    feat.extend([wg[0]/wgs, wg[1]/wgs])

    wn  = _wind_at(_predict_next_wind(wind_field), x, y)
    wns = np.linalg.norm(wn) + 1e-9
    feat.extend([wn[0]/wns, wn[1]/wns])

    for dx,dy in [(0,1),(1,1),(1,0),(0,-1),(-1,0),(-1,1)]:
        feat.append(_sailing_efficiency(np.array([dx,dy]), np.array([wx,wy])))

    feat.extend([np.arctan2(wy,wx)/np.pi, wind_speed/10.0, 0.0])

    for i in range(1, 7):
        t  = i/7.0
        wp = _wind_at(wind_field, x+t*gx, y+t*gy)
        feat.append(_sailing_efficiency(np.array([gx,gy]), wp))

    mid = GRID_SIZE//2
    for x0,x1,y0,y1 in [(0,mid,mid,127),(mid,127,mid,127),(0,mid,0,mid),(mid,127,0,mid)]:
        wq = _zone_mean_wind(wind_field, x0, x1, y0, y1)
        feat.append(np.arctan2(wq[1], wq[0])/np.pi)

    wl = _zone_mean_wind(wind_field, 0, xl-1, yb, yt)
    wr = _zone_mean_wind(wind_field, 64, xr+1, yb, yt)
    feat.append(_sailing_efficiency(np.array([0.,1.]), wl) - _sailing_efficiency(np.array([0.,-1.]), wr))
    feat.append(v_speed/8.0)
    dist_to_island = min(abs(x-xl), abs(x-xr), abs(y-yb), abs(y-yt))
    feat.append(float(dist_to_island < 10))

    return np.array(feat, dtype=np.float32)


N_FEATURES = 56


# ═══════════════════════════════════════════════════════════════════════════════
#  SHAPED REWARD  (identique à agent_ppo_bc.py)
# ═══════════════════════════════════════════════════════════════════════════════

def shaped_reward(obs_prev, obs_curr, env_reward, terminated, is_stuck, step):
    x_p, y_p = obs_prev[0], obs_prev[1]
    x_c, y_c = obs_curr[0], obs_curr[1]
    wx, wy   = obs_curr[4], obs_curr[5]
    vx, vy   = obs_curr[2], obs_curr[3]

    d_prev    = np.sqrt((GOAL_X-x_p)**2 + (GOAL_Y-y_p)**2)
    d_curr    = np.sqrt((GOAL_X-x_c)**2 + (GOAL_Y-y_c)**2)
    progress  = (d_prev - d_curr) / GRID_SIZE * 5.0
    collision = -50.0 if is_stuck else 0.0
    time_pen  = -0.02
    v_speed   = np.sqrt(vx**2+vy**2)
    eff_bonus = _sailing_efficiency(np.array([vx,vy]), np.array([wx,wy]))*0.05 if v_speed>0.1 else 0.0
    return env_reward + progress + collision + time_pen + eff_bonus


# ═══════════════════════════════════════════════════════════════════════════════
#  RÉSEAU ACTOR-CRITIC  (identique à agent_ppo_bc.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_network(nn_module):
    nn = nn_module
    class ActorCritic(nn.Module):
        def __init__(self, n_feat=N_FEATURES, n_actions=9, hidden=128):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(n_feat, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
            )
            self.actor_head  = nn.Linear(hidden, n_actions)
            self.critic_head = nn.Linear(hidden, 1)
            for layer in self.shared:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
            nn.init.zeros_(self.actor_head.bias)
            nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
            nn.init.zeros_(self.critic_head.bias)

        def forward(self, x):
            h = self.shared(x)
            return self.actor_head(h), self.critic_head(h).squeeze(-1)

        def get_action_and_value(self, x, action=None):
            logits, value = self(x)
            dist = Categorical(logits=logits)
            if action is None: action = dist.sample()
            return action, dist.log_prob(action), dist.entropy(), value

    return ActorCritic()


def _save_weights(model, path, metrics=None):
    weights = {n: p.cpu().numpy() for n, p in model.state_dict().items()}
    if metrics:
        weights['_metrics_json'] = np.array([json.dumps(metrics)])
    np.savez(path, **weights)


# ═══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATEUR DE SCÉNARIOS ALÉATOIRES
# ═══════════════════════════════════════════════════════════════════════════════

_FIXED_WIND_INIT = {'base_speed': 10.0, 'base_max_rotation_angle_degree': 10}
_FIXED_WIND_EVOL = {'mean_rotation_angle_degree': 3, 'std_rotation_angle_degree': 0.8}
ORIGINAL_SCENARIOS = ('training_1', 'training_2', 'training_3')


def generate_random_scenario(rng: np.random.Generator, easy: bool = False) -> dict:
    """
    Génère un scénario aléatoire avec cohérence spatiale.

    easy=True  : vent fort (amp > 0.7) avec une direction majoritairement
                 favorable au déplacement vers le goal (cap ~NE/N/NW).
                 Utilisé pour le curriculum en début d'entraînement.
    easy=False : toutes directions, amplitudes variées ∈ [0.4, 1.0].
    """
    # Direction de base globale
    if easy:
        # Directions favorables = vers le haut-droit de la carte (goal est en haut)
        # vent venant du bas/bas-gauche → pousse le bateau vers le haut
        base_angle = rng.uniform(np.pi * 0.8, np.pi * 1.4)   # ~sud à sud-ouest
        base_amp   = rng.uniform(0.7, 1.0)
        noise_std  = np.pi / 6   # ±30° : assez cohérent
    else:
        base_angle = rng.uniform(0, 2*np.pi)
        base_amp   = rng.uniform(0.4, 1.0)
        noise_std  = np.pi / 3   # ±60° : diversifié

    pattern = []
    for _ in range(3):
        row = []
        for _ in range(3):
            angle = base_angle + rng.normal(0, noise_std)
            amp   = float(np.clip(base_amp + rng.uniform(-0.3, 0.3), 0.35, 1.0))
            vx    = float(np.clip(np.round(amp * np.cos(angle), 2), -1.0, 1.0))
            vy    = float(np.clip(np.round(amp * np.sin(angle), 2), -1.0, 1.0))
            row.append((vx, vy))
        pattern.append(tuple(row))

    return {
        'wind_init_params': {**_FIXED_WIND_INIT, 'pattern': tuple(pattern)},
        'wind_evol_params': dict(_FIXED_WIND_EVOL),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPERT A* WIND-AWARE  (Dijkstra sur grille avec coût sailing)
# ═══════════════════════════════════════════════════════════════════════════════

# Voisins 8-connectés
_NEIGHBORS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
# Index d'action correspondant
_NEIGHBOR_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7]


def _build_cost_map(wind_field: np.ndarray, world_map: np.ndarray) -> np.ndarray:
    """
    Construit une carte de coût 128×128 : coût pour SE DÉPLACER vers chaque cellule.
    cost(x,y) = 1 / efficacité_voile(direction_vers_(x,y), vent_en_(x,y))
    Les cellules île ont coût infini.
    """
    cost = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for ay in range(GRID_SIZE):
        for ax in range(GRID_SIZE):
            if world_map[ay, ax] == 1:
                cost[ay, ax] = np.inf
    return cost


def _dijkstra_wind_aware(
    start_x: int, start_y: int,
    goal_x: int, goal_y: int,
    wind_field: np.ndarray,
    world_map: np.ndarray,
) -> np.ndarray:
    """
    Dijkstra wind-aware sur grille 128×128.

    Le coût d'un arc (x,y) → (nx,ny) est :
        1.0 / sailing_efficiency(direction (nx-x, ny-y), vent en (nx,ny))

    Retourne un tableau action_map[y,x] = meilleure action depuis (x,y).
    """
    INF = 1e18
    dist      = np.full((GRID_SIZE, GRID_SIZE), INF, dtype=np.float64)
    action_map= np.full((GRID_SIZE, GRID_SIZE), 8,   dtype=np.int8)   # 8 = stay

    dist[start_y, start_x] = 0.0
    # heap : (dist, x, y)
    heap = [(0.0, start_x, start_y)]

    while heap:
        d, cx, cy = heapq.heappop(heap)
        if d > dist[cy, cx]:
            continue
        if cx == goal_x and cy == goal_y:
            break
        for act_idx, (dx, dy) in enumerate(zip(
            [a[0] for a in _NEIGHBORS],
            [a[1] for a in _NEIGHBORS]
        )):
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                continue
            if world_map[ny, nx] == 1:
                continue
            w  = wind_field[ny, nx]
            eff = _sailing_efficiency(np.array([dx, dy], dtype=float), w)
            # coût = distance euclidienne / efficacité (privilégie les arcs efficaces)
            arc_cost = np.sqrt(dx**2 + dy**2) / max(eff, 0.01)
            nd = d + arc_cost
            if nd < dist[ny, nx]:
                dist[ny, nx] = nd
                action_map[ny, nx] = act_idx
                heapq.heappush(heap, (nd, nx, ny))

    return action_map


# Cache Dijkstra : on ne recalcule que si le champ de vent change significativement
class WindAwareExpert:
    """
    Expert basé sur Dijkstra wind-aware.

    Pour chaque nouvel épisode, on calcule la carte d'actions depuis le goal
    (Dijkstra depuis goal → toute la grille), puis act() lit la case courante.

    Le Dijkstra est coûteux (~50ms sur CPU), donc on le précalcule une fois
    par épisode en début d'épisode (appel à reset()).
    """

    def __init__(self):
        self._action_map = None   # (128,128) int8
        self._goal = (GOAL_X, GOAL_Y)

    def reset(self, obs_raw: np.ndarray) -> None:
        """Calcule la carte d'actions pour cet épisode."""
        wind_field = obs_raw[6:6+GRID_SIZE*GRID_SIZE*2].reshape(GRID_SIZE, GRID_SIZE, 2)
        world_map  = obs_raw[6+GRID_SIZE*GRID_SIZE*2:].reshape(GRID_SIZE, GRID_SIZE)
        gx, gy     = self._goal

        # Dijkstra depuis GOAL vers toute la grille
        # (on inverse : coût pour REJOINDRE le goal depuis n'importe quelle cellule)
        self._action_map = _dijkstra_from_goal(gx, gy, wind_field, world_map)

    def act(self, obs_raw: np.ndarray) -> int:
        if self._action_map is None:
            self.reset(obs_raw)
        px = int(np.clip(round(float(obs_raw[0])), 0, GRID_SIZE-1))
        py = int(np.clip(round(float(obs_raw[1])), 0, GRID_SIZE-1))
        return int(self._action_map[py, px])


def _dijkstra_from_goal(
    goal_x: int, goal_y: int,
    wind_field: np.ndarray,
    world_map:  np.ndarray,
) -> np.ndarray:
    """
    Dijkstra inversé : calcule pour CHAQUE cellule (x,y)
    la meilleure action pour se rapprocher du goal.

    Algorithme : Dijkstra depuis le goal en sens inverse.
    Pour chaque cellule prédécesseur (px,py) → (goal_x,goal_y) via (dx,dy) :
      - coût arc = euclidean(dx,dy) / sailing_eff(direction (dx,dy), vent en (goal_x,goal_y))
    On enregistre l'action qui depuis (px,py) va vers le prochain nœud optimal.
    """
    INF = 1e18
    dist       = np.full((GRID_SIZE, GRID_SIZE), INF, dtype=np.float64)
    action_map = np.full((GRID_SIZE, GRID_SIZE), 8,   dtype=np.int8)

    dist[goal_y, goal_x] = 0.0
    heap = [(0.0, goal_x, goal_y)]

    while heap:
        d, cx, cy = heapq.heappop(heap)
        if d > dist[cy, cx]:
            continue

        # Depuis (cx,cy), on remonte vers les prédécesseurs (px,py)
        # i.e. on cherche tous les (px,py) tels que (px→cx) est un mouvement valide
        for act_idx, (dx, dy) in enumerate(_NEIGHBORS):
            # Le prédécesseur serait px = cx-dx, py = cy-dy
            px, py = cx - dx, cy - dy
            if px < 0 or px >= GRID_SIZE or py < 0 or py >= GRID_SIZE:
                continue
            if world_map[py, px] == 1:
                continue
            # L'action depuis (px,py) pour aller vers (cx,cy) est act_idx
            # Le vent au point d'arrivée (cx,cy) détermine l'efficacité
            w   = wind_field[cy, cx]
            eff = _sailing_efficiency(np.array([dx, dy], dtype=float), w)
            arc = np.sqrt(dx**2 + dy**2) / max(eff, 0.01)
            nd  = d + arc
            if nd < dist[py, px]:
                dist[py, px]      = nd
                action_map[py, px] = act_idx   # action à prendre depuis (px,py)
                heapq.heappush(heap, (nd, px, py))

    return action_map


# ═══════════════════════════════════════════════════════════════════════════════
#  BUFFER BC  — collecte de trajectoires d'expert
# ═══════════════════════════════════════════════════════════════════════════════

class BCBuffer:
    """
    Buffer circulaire de transitions (features, expert_action).
    Rempli en parallèle pendant les rollouts PPO.
    """
    def __init__(self, capacity: int = 50_000):
        self.capacity = capacity
        self.feats    = np.zeros((capacity, N_FEATURES), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.ptr      = 0
        self.size     = 0

    def add(self, feat: np.ndarray, action: int):
        self.feats[self.ptr]   = feat
        self.actions[self.ptr] = action
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.feats[idx], self.actions[idx]

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════════
#  POOL D'ENVIRONNEMENTS AVEC CURRICULUM
# ═══════════════════════════════════════════════════════════════════════════════

class CurriculumEnvPool:
    """
    Gère n_envs environnements avec :
    - orig_ratio  : fraction sur scénarios originaux (anti-forgetting)
    - easy_ratio  : fraction sur scénarios faciles (curriculum phase 1)
    - rest        : scénarios aléatoires complets
    La difficulté augmente linéairement avec global_step / curriculum_steps.
    """

    def __init__(self, n_envs, orig_ratio, rng, SailingEnv, get_scenario):
        self.n_envs       = n_envs
        self.orig_ratio   = orig_ratio
        self.rng          = rng
        self.SailingEnv   = SailingEnv
        self.get_scenario = get_scenario

        self.n_orig   = max(1, int(round(n_envs * orig_ratio)))
        self.n_random = n_envs - self.n_orig

        self._easy_frac = 1.0   # décroît au fil du temps (mis à jour par le trainer)

        self.envs    = []
        self.raw_obs = []
        for i in range(n_envs):
            env, obs = self._make(i)
            self.envs.append(env)
            self.raw_obs.append(obs)

        print(f"[EnvPool] {n_envs} envs : {self.n_orig} originaux + {self.n_random} random")

    def set_easy_frac(self, frac: float):
        """frac ∈ [0,1] : 1 = tous faciles, 0 = tous random difficiles."""
        self._easy_frac = float(np.clip(frac, 0.0, 1.0))

    def _make(self, idx: int):
        if idx < self.n_orig:
            sc     = ORIGINAL_SCENARIOS[idx % len(ORIGINAL_SCENARIOS)]
            params = self.get_scenario(sc)
        else:
            # Curriculum : avec proba easy_frac on tire un scénario facile
            easy   = (self.rng.random() < self._easy_frac)
            params = generate_random_scenario(self.rng, easy=easy)

        env = self.SailingEnv(**params)
        env.seed(int(self.rng.integers(0, 2**31)))
        obs, _ = env.reset()
        return env, obs

    def reset_env(self, idx: int) -> np.ndarray:
        """Réinitialise avec un nouveau scénario (pour les envs random)."""
        if idx < self.n_orig:
            obs, _ = self.envs[idx].reset()
        else:
            env, obs = self._make(idx)
            self.envs[idx] = env
        self.raw_obs[idx] = obs
        return obs


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRAÎNEMENT PPO + BC INTERLEAVED + CURRICULUM + RANDOM SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    save_path             = "agent_ppo_bc_v2_weights.npz",
    weights_init          = None,       # chemin .npz à charger pour warm-start (optionnel)
    n_envs                = 20,
    total_steps           = 4_000_000,
    rollout_steps         = 512,
    n_epochs              = 4,
    minibatch_size        = 256,
    lr                    = 3e-4,
    gamma                 = 0.995,
    gae_lambda            = 0.95,
    clip_coef             = 0.2,
    vf_coef               = 0.5,
    ent_coef              = 0.01,
    max_grad_norm         = 0.5,
    # ── BC interleaved ──────────────────────────────────────────────────
    bc_coef_start         = 1.0,        # λ_bc initial (fort au début)
    bc_coef_end           = 0.05,       # λ_bc final (quasi-éteint en fin d'entraîn.)
    bc_decay_steps        = 1_500_000,  # steps sur lesquels λ_bc décroît
    bc_batch_size         = 256,        # taille mini-batch BC par update
    bc_buffer_capacity    = 80_000,     # transitions expert stockées
    bc_expert_prob        = 0.3,        # proba qu'un step collecte aussi l'action expert
    # ── Curriculum ──────────────────────────────────────────────────────
    curriculum_end_frac   = 0.25,       # fraction des steps où easy_frac passe de 1→0
    orig_ratio            = 0.15,       # fraction envs sur scénarios originaux
    # ── Divers ──────────────────────────────────────────────────────────
    log_interval          = 100,
    checkpoint_interval   = 200,
    device_str            = "auto",
    seed                  = 42,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch requis pour l'entraînement.")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from src.env_sailing import SailingEnv
        from src.wind_scenarios import get_wind_scenario
    except ImportError:
        from env_sailing import SailingEnv
        from wind_scenarios import get_wind_scenario

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    print(f"[PPO-BC v2] Device : {device}")
    print(f"[PPO-BC v2] Steps : {total_steps:,} | n_envs : {n_envs}")
    print(f"[PPO-BC v2] BC λ : {bc_coef_start} → {bc_coef_end} sur {bc_decay_steps:,} steps")
    print(f"[PPO-BC v2] Curriculum : easy→hard sur {curriculum_end_frac*100:.0f}% du training")
    print(f"[PPO-BC v2] orig_ratio : {orig_ratio:.0%}")

    # ── Modèle ────────────────────────────────────────────────────────────────
    model     = _make_network(nn).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    if weights_init and os.path.exists(weights_init):
        data = np.load(weights_init, allow_pickle=True)
        sd   = {k: torch.tensor(data[k]) for k in data.files if not k.startswith('_')}
        model.load_state_dict(sd, strict=True)
        print(f"[PPO-BC v2] Poids chargés depuis {weights_init}")

    bc_loss_fn = nn.CrossEntropyLoss()

    # ── Pool d'environnements ─────────────────────────────────────────────────
    pool = CurriculumEnvPool(n_envs, orig_ratio, rng, SailingEnv, get_wind_scenario)

    # ── Expert A* + BC buffer ─────────────────────────────────────────────────
    experts    = [WindAwareExpert() for _ in range(n_envs)]
    bc_buffer  = BCBuffer(capacity=bc_buffer_capacity)

    # Pré-calcul des cartes experts pour les envs initiaux
    print("[PPO-BC v2] Précalcul des cartes A* initiales...")
    t0 = time.time()
    for i in range(n_envs):
        experts[i].reset(pool.raw_obs[i])
    print(f"  → {time.time()-t0:.1f}s pour {n_envs} cartes")

    # ── Buffers PPO ───────────────────────────────────────────────────────────
    buf_obs      = torch.zeros(rollout_steps, n_envs, N_FEATURES).to(device)
    buf_actions  = torch.zeros(rollout_steps, n_envs, dtype=torch.long).to(device)
    buf_logprobs = torch.zeros(rollout_steps, n_envs).to(device)
    buf_rewards  = torch.zeros(rollout_steps, n_envs).to(device)
    buf_dones    = torch.zeros(rollout_steps, n_envs).to(device)
    buf_values   = torch.zeros(rollout_steps, n_envs).to(device)

    next_obs = torch.tensor(
        np.stack([extract_features(o) for o in pool.raw_obs]),
        dtype=torch.float32,
    ).to(device)
    next_dones = torch.zeros(n_envs).to(device)

    # ── Métriques ─────────────────────────────────────────────────────────────
    metrics = {
        'success_rate':        [],
        'collision_rate':      [],
        'mean_score':          [],
        'mean_steps_success':  [],
        'mean_shaped_reward':  [],
        'policy_loss':         [],
        'value_loss':          [],
        'entropy':             [],
        'approx_kl':           [],
        'bc_loss':             [],
        'bc_coef':             [],
        'easy_frac':           [],
        'n_random_generated':  [],
    }

    ep_rewards    = [0.0]  * n_envs
    ep_lengths    = [0]    * n_envs
    ep_successes  = [False]* n_envs
    ep_collisions = [False]* n_envs
    ep_shaped     = [0.0]  * n_envs

    recent_success   = deque(maxlen=200)
    recent_collision = deque(maxlen=200)
    recent_reward    = deque(maxlen=200)
    recent_length    = deque(maxlen=200)
    recent_shaped    = deque(maxlen=200)
    recent_bc_loss   = deque(maxlen=100)

    completed_episodes = 0
    n_random_generated = 0
    global_step        = 0
    n_updates          = total_steps // (n_envs * rollout_steps)
    curriculum_steps   = int(total_steps * curriculum_end_frac)
    t_start            = time.time()

    print(f"[PPO-BC v2] {n_updates} updates au total\n")

    for update in range(1, n_updates + 1):

        # ── Curriculum & λ_bc courants ────────────────────────────────────────
        easy_frac = max(0.0, 1.0 - global_step / max(curriculum_steps, 1))
        pool.set_easy_frac(easy_frac)

        bc_coef = bc_coef_end + (bc_coef_start - bc_coef_end) * max(
            0.0, 1.0 - global_step / max(bc_decay_steps, 1)
        )

        # Anneal LR linéaire
        lr_now = lr * (1.0 - (update - 1) / n_updates)
        for pg in optimizer.param_groups:
            pg['lr'] = lr_now

        # ── Collecte du rollout ───────────────────────────────────────────────
        for step in range(rollout_steps):
            global_step += n_envs
            buf_obs[step]   = next_obs
            buf_dones[step] = next_dones

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(next_obs)
            buf_actions[step]  = action
            buf_logprobs[step] = log_prob
            buf_values[step]   = value

            new_obs_list = []
            step_rewards = []
            step_dones   = []

            for i in range(n_envs):
                o_raw = pool.raw_obs[i]
                a     = action[i].item()

                # Collecte action expert dans le BC buffer (stochastique)
                if rng.random() < bc_expert_prob:
                    expert_a = experts[i].act(o_raw)
                    bc_buffer.add(extract_features(o_raw), expert_a)

                o_next, r, terminated, truncated, info = pool.envs[i].step(a)

                sr = shaped_reward(o_raw, o_next, r, terminated,
                                   info.get('is_stuck', False), ep_lengths[i])

                ep_rewards[i]   += r
                ep_lengths[i]   += 1
                ep_shaped[i]    += sr
                if info.get('is_stuck', False): ep_collisions[i] = True
                if r > 50:                      ep_successes[i]  = True

                done = terminated or truncated
                step_rewards.append(sr)
                step_dones.append(float(done))
                pool.raw_obs[i] = o_next

                if done:
                    recent_success.append(float(ep_successes[i]))
                    recent_collision.append(float(ep_collisions[i]))
                    recent_reward.append(ep_rewards[i])
                    recent_shaped.append(ep_shaped[i])
                    if ep_successes[i]:
                        recent_length.append(ep_lengths[i])

                    completed_episodes += 1

                    if completed_episodes % log_interval == 0:
                        elapsed = time.time() - t_start
                        bc_l    = float(np.mean(recent_bc_loss)) if recent_bc_loss else 0.0
                        print(
                            f"[Ep {completed_episodes:6d} | {global_step:8d} steps | "
                            f"{elapsed/60:.1f}min] "
                            f"Succès={np.mean(recent_success)*100:.1f}% | "
                            f"Collision={np.mean(recent_collision)*100:.1f}% | "
                            f"Score={np.mean(recent_reward):.2f} | "
                            f"Steps={np.mean(recent_length) if recent_length else 0:.1f} | "
                            f"SR={np.mean(recent_shaped):.2f} | "
                            f"BC_loss={bc_l:.4f} | λ_bc={bc_coef:.3f} | "
                            f"easy={easy_frac:.2f} | lr={lr_now:.2e}"
                        )
                        metrics['success_rate'].append(float(np.mean(recent_success)))
                        metrics['collision_rate'].append(float(np.mean(recent_collision)))
                        metrics['mean_score'].append(float(np.mean(recent_reward)))
                        metrics['mean_shaped_reward'].append(float(np.mean(recent_shaped)))
                        metrics['bc_loss'].append(bc_l)
                        metrics['bc_coef'].append(float(bc_coef))
                        metrics['easy_frac'].append(float(easy_frac))
                        metrics['n_random_generated'].append(n_random_generated)
                        if recent_length:
                            metrics['mean_steps_success'].append(float(np.mean(recent_length)))

                    if completed_episodes % checkpoint_interval == 0:
                        _save_weights(model, save_path, metrics)
                        print(f"  ✓ Checkpoint → {save_path}")

                    # Reset env + nouveau scénario + nouvelle carte A*
                    new_o = pool.reset_env(i)
                    if i >= pool.n_orig:
                        n_random_generated += 1
                    experts[i].reset(new_o)   # recalcul A* sur le nouveau scénario
                    o_next = new_o

                    ep_rewards[i]    = 0.0
                    ep_lengths[i]    = 0
                    ep_successes[i]  = False
                    ep_collisions[i] = False
                    ep_shaped[i]     = 0.0

                new_obs_list.append(extract_features(o_next))

            buf_rewards[step] = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            next_obs   = torch.tensor(np.stack(new_obs_list), dtype=torch.float32).to(device)
            next_dones = torch.tensor(step_dones,             dtype=torch.float32).to(device)

        # ── GAE ───────────────────────────────────────────────────────────────
        with torch.no_grad():
            nv   = model.get_action_and_value(next_obs)[3]
            advs = torch.zeros_like(buf_rewards)
            last = 0.0
            for t in reversed(range(rollout_steps)):
                nnt  = 1.0 - (next_dones if t == rollout_steps-1 else buf_dones[t+1])
                nvt  = nv if t == rollout_steps-1 else buf_values[t+1]
                delta = buf_rewards[t] + gamma * nvt * nnt - buf_values[t]
                last  = delta + gamma * gae_lambda * nnt * last
                advs[t] = last
            returns = advs + buf_values

        # ── Update PPO + BC interleaved ───────────────────────────────────────
        b_obs   = buf_obs.reshape(-1, N_FEATURES)
        b_act   = buf_actions.reshape(-1)
        b_lp    = buf_logprobs.reshape(-1)
        b_adv   = advs.reshape(-1)
        b_ret   = returns.reshape(-1)
        b_adv   = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        n_samples = rollout_steps * n_envs
        inds      = np.arange(n_samples)
        upg, uvf, uent, ukl, ubc = [], [], [], [], []

        for epoch in range(n_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_samples, minibatch_size):
                mb = inds[start:start + minibatch_size]

                # ── Gradient PPO ─────────────────────────────────────────────
                _, new_lp, entropy, new_val = model.get_action_and_value(b_obs[mb], b_act[mb])
                log_ratio  = new_lp - b_lp[mb]
                ratio      = log_ratio.exp()
                approx_kl  = ((ratio - 1) - log_ratio).mean().item()
                mb_adv     = b_adv[mb]

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1-clip_coef, 1+clip_coef),
                ).mean()
                vf_loss  = 0.5 * (new_val - b_ret[mb]).pow(2).mean()
                ent_loss = entropy.mean()
                ppo_loss = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

                # ── Gradient BC (si buffer assez rempli) ────────────────────
                bc_loss_val = torch.tensor(0.0, device=device)
                if len(bc_buffer) >= bc_batch_size and bc_coef > 1e-4:
                    bc_feats, bc_acts = bc_buffer.sample(bc_batch_size)
                    bc_ft   = torch.tensor(bc_feats, dtype=torch.float32).to(device)
                    bc_at   = torch.tensor(bc_acts,  dtype=torch.long).to(device)
                    bc_logits, _ = model(bc_ft)
                    bc_loss_val  = bc_loss_fn(bc_logits, bc_at)

                total_loss = ppo_loss + bc_coef * bc_loss_val

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                upg.append(pg_loss.item())
                uvf.append(vf_loss.item())
                uent.append(ent_loss.item())
                ukl.append(approx_kl)
                if bc_loss_val.item() > 0:
                    ubc.append(bc_loss_val.item())
                    recent_bc_loss.append(bc_loss_val.item())

        metrics['policy_loss'].append(float(np.mean(upg)))
        metrics['value_loss'].append(float(np.mean(uvf)))
        metrics['entropy'].append(float(np.mean(uent)))
        metrics['approx_kl'].append(float(np.mean(ukl)))

    # ── Sauvegarde finale ─────────────────────────────────────────────────────
    _save_weights(model, save_path, metrics)
    elapsed = time.time() - t_start
    print(f"\n[PPO-BC v2] Terminé en {elapsed/60:.1f} min")
    print(f"  Scénarios aléatoires : {n_random_generated:,}")
    print(f"  Taux de succès final  : {np.mean(list(recent_success))*100:.1f}%")
    print(f"  Taux de collision fin : {np.mean(list(recent_collision))*100:.1f}%")
    print(f"  Score moyen final     : {np.mean(list(recent_reward)):.3f}")
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  INFÉRENCE NUMPY-ONLY  (identique à agent_ppo_bc.py)
# ═══════════════════════════════════════════════════════════════════════════════

class NumpyActorCritic:
    def __init__(self, weights: dict):
        self.W0 = weights['shared.0.weight']
        self.b0 = weights['shared.0.bias']
        self.W2 = weights['shared.2.weight']
        self.b2 = weights['shared.2.bias']
        self.Wa = weights['actor_head.weight']
        self.ba = weights['actor_head.bias']

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W0 @ x + self.b0)
        h = np.tanh(self.W2 @ h + self.b2)
        logits = self.Wa @ h + self.ba
        ex = np.exp(logits - logits.max())
        return ex / ex.sum()

    def act_greedy(self, x: np.ndarray) -> int:
        return int(np.argmax(self.forward(x)))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSE AGENT
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from agents.base_agent import BaseAgent
except ImportError:
    try:
        from base_agent import BaseAgent
    except ImportError:
        from src.agents.base_agent import BaseAgent


class MyAgent(BaseAgent):
    DEFAULT_WEIGHTS_PATH = "agent_ppo_bc_v2_weights.npz"

    def __init__(self, weights_path: str = None):
        super().__init__()
        self.np_random = np.random.default_rng()
        self._net: NumpyActorCritic = None
        path = weights_path or self.DEFAULT_WEIGHTS_PATH
        if os.path.exists(path):
            self.load(path)

    def act(self, observation: np.ndarray) -> int:
        feat = extract_features(observation)
        return self._net.act_greedy(feat) if self._net else self._fallback(observation)

    def _fallback(self, obs):
        x, y = int(obs[0]), int(obs[1])
        xl, xr, yb, yt = ISLAND_RECT
        if y < yb and x > xl-5 and x < xr+5:
            return 2 if x < GOAL_X else 6
        gx, gy = GOAL_X-x, GOAL_Y-y
        if abs(gx)<2 and gy>0: return 0
        if gx>0 and gy>0:       return 1
        if gx>0:                 return 2
        if gx<0 and gy>0:       return 7
        if gx<0:                 return 6
        return 0

    def reset(self): pass
    def seed(self, seed=None): self.np_random = np.random.default_rng(seed)

    def save(self, path: str):
        if self._net:
            np.savez(path,
                **{'shared.0.weight': self._net.W0, 'shared.0.bias': self._net.b0,
                   'shared.2.weight': self._net.W2, 'shared.2.bias': self._net.b2,
                   'actor_head.weight': self._net.Wa, 'actor_head.bias': self._net.ba})
            print(f"[MyAgent] Sauvegardé → {path}")

    def load(self, path: str):
        try:
            data = np.load(path, allow_pickle=True)
            self._net = NumpyActorCritic({k: data[k] for k in data.files if not k.startswith('_')})
            print(f"[MyAgent] Poids chargés depuis {path}")
        except Exception as e:
            print(f"[MyAgent] Erreur chargement {path} : {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(weights_path="agent_ppo_bc_v2_weights.npz", n_episodes=50, seed_offset=9999):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from src.env_sailing import SailingEnv
        from src.wind_scenarios import get_wind_scenario
    except ImportError:
        from env_sailing import SailingEnv
        from wind_scenarios import get_wind_scenario

    agent   = MyAgent(weights_path=weights_path)
    results = {sc: {'success':0,'collision':0,'rewards':[],'lengths':[]}
               for sc in ORIGINAL_SCENARIOS}
    total_s = total_c = 0
    all_r   = []

    for ep in range(n_episodes):
        sc     = ORIGINAL_SCENARIOS[ep % len(ORIGINAL_SCENARIOS)]
        env    = SailingEnv(**get_wind_scenario(sc))
        env.seed(seed_offset + ep)
        obs, _ = env.reset()
        agent.reset()
        done = False; ep_r = 0.0; disc = 1.0; steps = 0

        while not done:
            obs, r, terminated, truncated, info = env.step(agent.act(obs))
            ep_r += disc * r; disc *= 0.995; steps += 1
            done = terminated or truncated

        s = ep_r > 50; c = info.get('is_stuck', False)
        results[sc]['rewards'].append(ep_r)
        results[sc]['success']   += int(s)
        results[sc]['collision'] += int(c)
        if s: results[sc]['lengths'].append(steps)
        total_s += int(s); total_c += int(c); all_r.append(ep_r)

    print("\n" + "═"*60)
    print("RÉSULTATS D'ÉVALUATION — PPO-BC v2")
    print("═"*60)
    print(f"  Épisodes       : {n_episodes}")
    print(f"  Taux succès    : {total_s/n_episodes*100:.1f}%")
    print(f"  Taux collision : {total_c/n_episodes*100:.1f}%")
    print(f"  Score moyen    : {np.mean(all_r):.3f}")
    all_len = [l for sc in ORIGINAL_SCENARIOS for l in results[sc]['lengths']]
    if all_len: print(f"  Steps (succès) : {np.mean(all_len):.1f}")
    print()
    for sc in ORIGINAL_SCENARIOS:
        r = results[sc]; n = n_episodes//len(ORIGINAL_SCENARIOS)
        print(f"  [{sc}] Succès={r['success']}/{n} | Collision={r['collision']} "
              f"| Score={np.mean(r['rewards']):.3f}"
              + (f" | Steps={np.mean(r['lengths']):.1f}" if r['lengths'] else ""))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Import s3fs si disponible (infra SSPCloud)
    try:
        import s3fs
        fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': 'https://minio.lab.sspcloud.fr'},
            key=os.environ.get("AWS_ACCESS_KEY_ID",""),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY",""),
            token=os.environ.get("AWS_SESSION_TOKEN",""),
        )
        S3_PREFIX = "mamorin/rl_sailing/models/"
        HAS_S3    = True
    except Exception:
        HAS_S3 = False

    parser = argparse.ArgumentParser(
        description="PPO-BC v2 — Expert A* + BC Interleaved + Curriculum + Random Scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train",           action="store_true")
    parser.add_argument("--eval",            action="store_true")
    parser.add_argument("--weights",         type=str,   default="agent_ppo_bc_v2_weights.npz")
    parser.add_argument("--weights_init",    type=str,   default=None,
                        help="Poids initiaux à charger avant entraînement (warm-start optionnel)")
    parser.add_argument("--steps",           type=int,   default=4_000_000)
    parser.add_argument("--n_envs",          type=int,   default=20)
    parser.add_argument("--n_eval",          type=int,   default=50)
    parser.add_argument("--device",          type=str,   default="auto")
    parser.add_argument("--rollout",         type=int,   default=512)
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--ent_coef",        type=float, default=0.01)
    parser.add_argument("--bc_coef_start",   type=float, default=1.0)
    parser.add_argument("--bc_coef_end",     type=float, default=0.05)
    parser.add_argument("--bc_decay_steps",  type=int,   default=1_500_000)
    parser.add_argument("--bc_expert_prob",  type=float, default=0.3,
                        help="Proba de collecter l'action expert à chaque step (0→1)")
    parser.add_argument("--curriculum_frac", type=float, default=0.25,
                        help="Fraction des steps pour transition easy→hard")
    parser.add_argument("--orig_ratio",      type=float, default=0.15)
    parser.add_argument("--log_interval",    type=int,   default=100)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    if not args.train and not args.eval:
        parser.print_help()
        sys.exit(0)

    if args.train:
        print("=" * 65)
        print("ENTRAÎNEMENT PPO-BC v2 — Expert A* + BC Interleaved + Curriculum")
        print("=" * 65)
        metrics = train(
            save_path           = args.weights,
            weights_init        = args.weights_init,
            total_steps         = args.steps,
            n_envs              = args.n_envs,
            rollout_steps       = args.rollout,
            lr                  = args.lr,
            ent_coef            = args.ent_coef,
            bc_coef_start       = args.bc_coef_start,
            bc_coef_end         = args.bc_coef_end,
            bc_decay_steps      = args.bc_decay_steps,
            bc_expert_prob      = args.bc_expert_prob,
            curriculum_end_frac = args.curriculum_frac,
            orig_ratio          = args.orig_ratio,
            log_interval        = args.log_interval,
            device_str          = args.device,
            seed                = args.seed,
        )
        metrics_path = args.weights.replace(".npz", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Métriques → {metrics_path}")

        if HAS_S3:
            for p in [args.weights, metrics_path]:
                if os.path.exists(p):
                    with open(p, 'rb') as fi, fs.open(S3_PREFIX+p, 'wb') as fo:
                        fo.write(fi.read())
                    print(f"Uploadé S3 → {S3_PREFIX+p}")

    if args.eval:
        print("=" * 65)
        print("ÉVALUATION — PPO-BC v2")
        print("=" * 65)
        evaluate(weights_path=args.weights, n_episodes=args.n_eval)