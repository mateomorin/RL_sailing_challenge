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

import base64
import zlib
import io
import numpy as np

from evaluator.base_agent import BaseAgent

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
#  INFÉRENCE EXPERT si fallback
# ═══════════════════════════════════════════════════════════════════════════════

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

    def __init__(self):
        super().__init__()
        self.np_random = np.random.default_rng()
        self._net: NumpyActorCritic = None

        compressed_bytes = base64.b64decode(WEIGHTS_B64)
        npz_bytes = zlib.decompress(compressed_bytes)
        with io.BytesIO(npz_bytes) as f:
            data = np.load(f, allow_pickle=True)
            weights = {k: data[k] for k in data.files if not k.startswith('_')}
        self._net = NumpyActorCritic(weights)
        print("[MyAgent] Succès : Les poids du réseau ont été chargés depuis la mémoire (Base64) !")

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
        """Retour au baseline si les poids ne sont pas disponibles."""
        return get_expert_action(obs_raw=obs)

    def reset(self) -> None:
        """Reset l'état interne de l'agent entre les épisodes."""
        pass

    def seed(self, seed: int = None) -> None:
        self.np_random = np.random.default_rng(seed)
