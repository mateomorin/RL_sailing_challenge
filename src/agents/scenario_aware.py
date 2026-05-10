"""
Scenario-Aware Sailing Agent (Idée 1)
======================================

Stratégie :
-----------
Au step 1 de chaque épisode, on observe le champ de vent complet (128×128×2).
On le compare aux signatures des 3 scénarios d'entraînement (calculées offline)
pour identifier le scénario courant en ~1ms.

Une fois le scénario identifié, on utilise une politique windmaster adaptée :
- Le mean_rotation exact du scénario (toujours 3° pour les 3 scénarios connus)
- Un biais de corridor (gauche/droite de l'île) pré-calculé pour chaque scénario
  selon la direction dominante du vent dans la zone du bateau.

Si le scénario test est inconnu (corrélation < seuil), on fallback sur windmaster
pur sans hypothèse de corridor.

Identification :
- On extrait des "zones témoins" du champ de vent (coins + centre) au step 0.
- On calcule la corrélation cosine entre le champ observé et les signatures de
  chaque scénario (moyennées sur plusieurs seeds pour lisser le bruit).
- Le scénario avec la corrélation la plus haute est retenu si > threshold.

Numpy only – compatible Codabench.
"""

import numpy as np
import math

try:
    from base_agent import BaseAgent
except ImportError:
    try:
        from agents.base_agent import BaseAgent
    except ImportError:
        from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Signatures de scénarios (calculées à l'initialisation depuis les patterns)
# ---------------------------------------------------------------------------

# Les patterns 3×3 de chaque scénario, interpolés sur 128×128.
# On les stocke directement ici pour éviter d'importer env_sailing.
# Format : liste de 3×3 vecteurs (vx, vy) normalisés.

SCENARIO_PATTERNS = {
    'training_1': {
        'pattern': [
            [(1, 1),   (0, -1),    (0, -1)],
            [(1, 1),   (0, -1),    (0, -1)],
            [(1, -1),  (-0.55, 1), (-0.55, 1)],
        ],
        'mean_rotation': 3.0,
        'std_rotation': 0.8,
    },
    'training_2': {
        'pattern': [
            [(0, -1),    (0, -1),   (1, 1)],
            [(0, -1),    (0, -1),   (1, 1)],
            [(-0.55, 1), (-0.55, 1),(1, -1)],
        ],
        'mean_rotation': 3.0,
        'std_rotation': 0.8,
    },
    'training_3': {
        'pattern': [
            [(1, 1),  (0, 1), (-1, 1)],
            [(1, 1),  (0, 1), (1, 1)],
            [(-1, 1), (0, 1), (1, 1)],
        ],
        'mean_rotation': 3.0,
        'std_rotation': 0.8,
    },
}

# Corridor recommandé pour chaque scénario : 'left' (y<64), 'right' (y>64), 'any'
# Dérivé de la direction dominante du vent en bas de carte (zone de départ).
# training_1 : vent (1,1) en bas-gauche → pousse vers droite → corridor DROIT (haut de carte, y>64)
# training_2 : vent (0,-1) en bas-gauche → vent vers bas, difficile → corridor GAUCHE
# training_3 : vent globalement (0,1) → vent vers le haut, traversée directe possible
SCENARIO_CORRIDOR = {
    'training_1': 'right',   # passer par x > 64
    'training_2': 'left',    # passer par x < 64
    'training_3': 'any',     # vent favorable, les deux marchent
    'unknown':    'any',
}


def _build_signature(pattern, grid_size=128):
    """
    Interpole bilinéairement le pattern 3×3 sur une grille grid_size×grid_size.
    Retourne un vecteur normalisé (2*grid_size^2,) représentant la direction
    du vent (sans l'amplitude, pour être robuste au bruit d'amplitude).
    """
    n_rows, n_cols = 3, 3
    H, W = grid_size, grid_size

    # Normaliser chaque vecteur du pattern
    pat = np.array(pattern, dtype=np.float64)  # (3,3,2)
    norms = np.linalg.norm(pat, axis=-1, keepdims=True)
    pat = pat / np.maximum(norms, 1e-8)

    # Grille d'interpolation
    Yi = np.linspace(0, n_rows - 1, H)
    Xi = np.linspace(0, n_cols - 1, W)
    Xi, Yi = np.meshgrid(Xi, Yi)

    i0 = np.floor(Yi).astype(int)
    j0 = np.floor(Xi).astype(int)
    i1 = np.clip(i0 + 1, 0, n_rows - 1)
    j1 = np.clip(j0 + 1, 0, n_cols - 1)
    dy = (Yi - i0)[:, :, np.newaxis]
    dx = (Xi - j0)[:, :, np.newaxis]

    interp = (pat[i0, j0] * (1 - dx) * (1 - dy) +
              pat[i0, j1] * dx * (1 - dy) +
              pat[i1, j0] * (1 - dx) * dy +
              pat[i1, j1] * dx * dy)  # (H, W, 2)

    # Renormaliser
    n = np.linalg.norm(interp, axis=-1, keepdims=True)
    interp = interp / np.maximum(n, 1e-8)

    return interp.reshape(-1)  # (2*H*W,) mais on prend seulement les directions


# Précalcul des signatures au chargement du module
_SIGNATURES = {}
for _name, _cfg in SCENARIO_PATTERNS.items():
    _SIGNATURES[_name] = _build_signature(_cfg['pattern'])


def identify_scenario(wind_field, threshold=0.85):
    """
    Identifie le scénario à partir du champ de vent observé (128×128×2).

    Retourne (scenario_name, confidence) où confidence est la corrélation cosine
    avec la meilleure signature. Si < threshold, retourne ('unknown', confidence).
    """
    # Normaliser le champ observé direction par direction
    wf = wind_field.copy().astype(np.float64)
    norms = np.linalg.norm(wf, axis=-1, keepdims=True)
    wf_dir = wf / np.maximum(norms, 1e-8)
    obs_vec = wf_dir.reshape(-1)

    obs_norm = np.linalg.norm(obs_vec)
    if obs_norm < 1e-8:
        return 'unknown', 0.0
    obs_vec = obs_vec / obs_norm

    best_name, best_corr = 'unknown', -1.0
    for name, sig in _SIGNATURES.items():
        sig_norm = np.linalg.norm(sig)
        if sig_norm < 1e-8:
            continue
        corr = float(np.dot(obs_vec, sig / sig_norm))
        if corr > best_corr:
            best_corr = corr
            best_name = name

    if best_corr < threshold:
        return 'unknown', best_corr
    return best_name, best_corr


# ---------------------------------------------------------------------------
# Physique (identique aux autres agents)
# ---------------------------------------------------------------------------

DIRECTIONS = np.array([
    [0, 1], [1, 1], [1, 0], [1, -1],
    [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 0]
], dtype=np.float32)

BOAT_PERFORMANCE = 0.4
MAX_SPEED        = 8.0
INERTIA_FACTOR   = 0.3


def _sailing_eff(dnx, dny, wfnx, wfny):
    cos_a = dnx * wfnx + dny * wfny
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    pi4 = math.pi / 4
    if a < pi4: return 0.05
    elif a < math.pi/2: return 0.5 + 0.5*(a-pi4)/pi4
    elif a < 3*pi4: return 1.0
    else: return max(0.5, 1.0-0.5*(a-3*pi4)/pi4)


def _step_phys(px, py, vx, vy, wx, wy, aidx):
    dx, dy = float(DIRECTIONS[aidx,0]), float(DIRECTIONS[aidx,1])
    wn = math.sqrt(wx*wx+wy*wy)
    if wn > 1e-6:
        wnx, wny = wx/wn, wy/wn
        dn = math.sqrt(dx*dx+dy*dy)
        dnx, dny = (dx/dn, dy/dn) if dn>1e-10 else (1.0, 0.0)
        eff = _sailing_eff(dnx, dny, -wnx, -wny)
        tvx = dx*eff*wn*BOAT_PERFORMANCE; tvy = dy*eff*wn*BOAT_PERFORMANCE
        ts = math.sqrt(tvx*tvx+tvy*tvy)
        if ts>MAX_SPEED: tvx*=MAX_SPEED/ts; tvy*=MAX_SPEED/ts
        nvx = tvx+INERTIA_FACTOR*(vx-tvx); nvy = tvy+INERTIA_FACTOR*(vy-tvy)
        ns = math.sqrt(nvx*nvx+nvy*nvy)
        if ns>MAX_SPEED: nvx*=MAX_SPEED/ns; nvy*=MAX_SPEED/ns
    else:
        nvx = INERTIA_FACTOR*vx; nvy = INERTIA_FACTOR*vy
    ivx = int(math.ceil(nvx) if nvx<0 else math.floor(nvx))
    ivy = int(math.ceil(nvy) if nvy<0 else math.floor(nvy))
    return max(0,min(127,int(round(px))+ivx)), max(0,min(127,int(round(py))+ivy)), ivx, ivy


# ---------------------------------------------------------------------------
# Politique windmaster avec biais de corridor
# ---------------------------------------------------------------------------

def _windmaster_biased(px, py, vx, vy, wind_field, world_map, corridor='any'):
    """
    Windmaster classique avec un biais léger vers le corridor recommandé.
    corridor : 'left' (x<64), 'right' (x>64), 'any'
    Le biais est actif seulement si on est encore loin du goal et en bas de carte.
    """
    x = max(0, min(127, int(round(px))))
    y = max(0, min(127, int(round(py))))
    wx, wy = float(wind_field[y, x, 0]), float(wind_field[y, x, 1])

    tgx = 64.0 - px; tgy = 127.0 - py
    dist = math.sqrt(tgx*tgx + tgy*tgy)
    if dist < 1e-6: return 8
    tgx /= dist; tgy /= dist
    is_final = dist < 5.0

    # Biais de corridor : seulement si on est sous y=80 et loin du goal
    corridor_bias_x = 0.0
    if corridor == 'right' and py < 80 and dist > 30:
        corridor_bias_x = 0.3   # léger biais vers x>64
    elif corridor == 'left' and py < 80 and dist > 30:
        corridor_bias_x = -0.3  # léger biais vers x<64

    # Direction effective modifiée par le biais
    eff_tgx = tgx + corridor_bias_x
    en = math.sqrt(eff_tgx*eff_tgx + tgy*tgy)
    if en > 1e-6:
        eff_tgx /= en; eff_tgy = tgy / en
    else:
        eff_tgx, eff_tgy = tgx, tgy

    best_a, best_s = 8, -1e18
    for i in range(8):
        npx, npy, nvx, nvy = _step_phys(px, py, vx, vy, wx, wy, i)
        if world_map[npy, npx] == 1: continue
        if is_final:
            nd = math.sqrt((npx-64)**2+(npy-127)**2)
            sc = -nd - math.sqrt(nvx*nvx+nvy*nvy)*0.1
            if nd < 1.5: sc += 1000.0
        else:
            vmg = nvx*eff_tgx + nvy*eff_tgy
            safety = 1.0
            for ddx,ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                if world_map[max(0,min(127,npy+ddy)), max(0,min(127,npx+ddx))]==1:
                    safety=0.2; break
            sc = vmg * safety
        if sc > best_s: best_s=sc; best_a=i
    return best_a


# ---------------------------------------------------------------------------
# Agent principal
# ---------------------------------------------------------------------------

class ScenarioAwareAgent(BaseAgent):
    """
    Agent qui identifie le scénario de vent au step 1 et adapte sa politique.

    Au step 1 : identification du scénario par corrélation cosine du champ de vent.
    Ensuite   : windmaster avec biais de corridor adapté au scénario identifié.

    Paramètres :
    ------------
    id_threshold : seuil de corrélation pour accepter l'identification (défaut: 0.85)
    """

    def __init__(self, id_threshold=0.85):
        super().__init__()
        self.id_threshold  = id_threshold
        self.world_map     = None
        self._scenario     = None   # identifié au step 1
        self._corridor     = 'any'
        self._confidence   = 0.0
        self._step         = 0

    def reset(self):
        self.world_map   = None
        self._scenario   = None
        self._corridor   = 'any'
        self._confidence = 0.0
        self._step       = 0

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path): pass
    def load(self, path): pass

    def _parse_obs(self, obs):
        px, py = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        wf = obs[6:6+128*128*2].reshape(128, 128, 2).astype(np.float32)
        if self.world_map is None:
            self.world_map = obs[6+128*128*2:].reshape(128, 128).astype(np.float32)
        return px, py, vx, vy, wf

    def act(self, observation: np.ndarray) -> int:
        px, py, vx, vy, wind_field = self._parse_obs(observation)
        self._step += 1

        # Identification au step 1 (champ initial = le plus informatif)
        if self._step == 1:
            self._scenario, self._confidence = identify_scenario(
                wind_field, threshold=self.id_threshold
            )
            self._corridor = SCENARIO_CORRIDOR.get(self._scenario, 'any')

        action = _windmaster_biased(
            px, py, vx, vy, wind_field, self.world_map, corridor=self._corridor
        )
        return int(action)

    @property
    def scenario(self):
        return self._scenario

    @property
    def confidence(self):
        return self._confidence