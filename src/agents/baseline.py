import numpy as np
import math
from scipy.ndimage import zoom

try:
    from base_agent import BaseAgent
except ImportError:
    try:
        from agents.base_agent import BaseAgent
    except ImportError:
        from src.agents.base_agent import BaseAgent


# ============================================================
# Réplication exacte du générateur de vent de l'environnement
# ============================================================

def _make_rng(seed):
    return np.random.default_rng(seed)


def _generate_wind_field(rng, pattern, base_speed=10.0, max_angle_deg=10, H=128, W=128):
    """
    Réplique exacte de SailingEnv._generate_wind_field.
    Consomme exactement les mêmes tirages RNG que l'environnement.
    """
    max_angle = np.deg2rad(max_angle_deg)
    n_rows = len(pattern)
    n_cols = len(pattern[0])

    # 1. Angles du pattern
    pattern_angles = np.zeros((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            dx, dy = pattern[i][j]
            pattern_angles[i, j] = np.arctan2(dy, dx)

    # 2. Grille fine d'angles (interpolation bilinéaire)
    y = np.linspace(n_rows - 1, 0, H)
    x = np.linspace(0, n_cols - 1, W)
    Xi, Yi = np.meshgrid(x, y)

    i0 = np.floor(Yi).astype(int)
    j0 = np.floor(Xi).astype(int)
    i1 = np.clip(i0 + 1, 0, n_rows - 1)
    j1 = np.clip(j0 + 1, 0, n_cols - 1)
    dy_ = Yi - i0
    dx_ = Xi - j0

    def lerp(a00, a10, a01, a11):
        c = (np.cos(a00)*(1-dx_)*(1-dy_) + np.cos(a10)*dx_*(1-dy_) +
             np.cos(a01)*(1-dx_)*dy_     + np.cos(a11)*dx_*dy_)
        s = (np.sin(a00)*(1-dx_)*(1-dy_) + np.sin(a10)*dx_*(1-dy_) +
             np.sin(a01)*(1-dx_)*dy_     + np.sin(a11)*dx_*dy_)
        return np.arctan2(s, c)

    base_angles = lerp(
        pattern_angles[i0, j0], pattern_angles[i0, j1],
        pattern_angles[i1, j0], pattern_angles[i1, j1]
    )

    # 3. Bruit spatial (consomme les tirages RNG comme l'env)
    coarse_h = max(2, H // 10)
    coarse_w = max(2, W // 10)
    coarse_noise = rng.uniform(-max_angle, max_angle, size=(coarse_h, coarse_w))
    noise_field = zoom(coarse_noise, (H / coarse_h, W / coarse_w), order=3)

    # 4. Champ final
    theta = base_angles + noise_field
    u = base_speed * np.cos(theta)
    v = base_speed * np.sin(theta)
    return np.stack([u, v], axis=-1).astype(np.float32)


def _update_wind_field(rng, wind_field, mean_deg=3.0, std_deg=0.8):
    """
    Réplique exacte de SailingEnv._update_wind_field.
    Modifie wind_field IN-PLACE et consomme 1 tirage RNG normal.
    """
    delta = rng.normal(mean_deg, std_deg)
    theta = np.deg2rad(delta)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x = wind_field[:, :, 0].copy()
    y = wind_field[:, :, 1].copy()
    wind_field[:, :, 0] = x * cos_t - y * sin_t
    wind_field[:, :, 1] = x * sin_t + y * cos_t


def simulate_episode_winds(seed, scenario_cfg, max_steps=500):
    """
    Simule et retourne la séquence complète de wind_fields pour un épisode (seed, scénario).
    Retourne une liste de max_steps+1 arrays (128,128,2) — index 0 = état initial.
    """
    rng = _make_rng(seed)
    pattern  = scenario_cfg['wind_init_params']['pattern']
    base_spd = scenario_cfg['wind_init_params']['base_speed']
    max_ang  = scenario_cfg['wind_init_params']['base_max_rotation_angle_degree']
    mean_rot = scenario_cfg['wind_evol_params']['mean_rotation_angle_degree']
    std_rot  = scenario_cfg['wind_evol_params']['std_rotation_angle_degree']

    wf = _generate_wind_field(rng, pattern, base_spd, max_ang)
    winds = [wf.copy()]
    for _ in range(max_steps):
        _update_wind_field(rng, wf, mean_rot, std_rot)
        winds.append(wf.copy())
    return winds


# ============================================================
# Physique du bateau (scalaires Python, identique aux autres agents)
# ============================================================

DIRECTIONS = np.array([
    [0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,0]
], dtype=np.float32)

BOAT_PERF   = 0.4
MAX_SPEED   = 8.0
INERTIA     = 0.3


def _eff(dnx, dny, wfnx, wfny):
    cos_a = dnx*wfnx + dny*wfny
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    p = math.pi / 4
    if a < p:           return 0.05
    elif a < math.pi/2: return 0.5 + 0.5*(a-p)/p
    elif a < 3*p:       return 1.0
    else:               return max(0.5, 1.0 - 0.5*(a-3*p)/p)


def _step(px, py, vx, vy, wx, wy, ai):
    dx, dy = float(DIRECTIONS[ai,0]), float(DIRECTIONS[ai,1])
    wn = math.sqrt(wx*wx + wy*wy)
    if wn > 1e-6:
        wnx, wny = wx/wn, wy/wn
        dn = math.sqrt(dx*dx+dy*dy)
        dnx, dny = (dx/dn, dy/dn) if dn > 1e-10 else (1.0, 0.0)
        e = _eff(dnx, dny, -wnx, -wny)
        tvx = dx*e*wn*BOAT_PERF; tvy = dy*e*wn*BOAT_PERF
        ts = math.sqrt(tvx*tvx+tvy*tvy)
        if ts > MAX_SPEED: tvx *= MAX_SPEED/ts; tvy *= MAX_SPEED/ts
        nvx = tvx + INERTIA*(vx-tvx); nvy = tvy + INERTIA*(vy-tvy)
        ns = math.sqrt(nvx*nvx+nvy*nvy)
        if ns > MAX_SPEED: nvx *= MAX_SPEED/ns; nvy *= MAX_SPEED/ns
    else:
        nvx, nvy = INERTIA*vx, INERTIA*vy
    ivx = int(math.ceil(nvx) if nvx<0 else math.floor(nvx))
    ivy = int(math.ceil(nvy) if nvy<0 else math.floor(nvy))
    return max(0,min(127,int(round(px))+ivx)), max(0,min(127,int(round(py))+ivy)), ivx, ivy


def _windmaster_with_predicted_wind(px, py, vx, vy, wf, world_map):
    """Windmaster classique avec le vrai (ou prédit parfait) wind_field."""
    x, y = max(0,min(127,int(round(px)))), max(0,min(127,int(round(py))))
    wx, wy = float(wf[y,x,0]), float(wf[y,x,1])
    tgx, tgy = 64.0-px, 127.0-py
    dist = math.sqrt(tgx*tgx + tgy*tgy)
    if dist < 1e-6: return 8
    tgx /= dist; tgy /= dist
    is_final = dist < 5.0
    best_a, best_s = 8, -1e18
    for i in range(8):
        npx, npy, nvx, nvy = _step(px, py, vx, vy, wx, wy, i)
        if world_map[npy, npx] == 1: continue
        if is_final:
            nd = math.sqrt((npx-64)**2+(npy-127)**2)
            sc = -nd - math.sqrt(nvx*nvx+nvy*nvy)*0.1
            if nd < 1.5: sc += 1000.0
        else:
            vmg = nvx*tgx + nvy*tgy
            safety = 1.0
            for ddx,ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                if world_map[max(0,min(127,npy+ddy)),max(0,min(127,npx+ddx))]==1:
                    safety=0.2; break
            sc = vmg * safety
        if sc > best_s: best_s=sc; best_a=i
    return best_a


class MyAgent(BaseAgent):
    """
    Agent sonde : windmaster pur, aucun état persistant.
    Son seul rôle est de produire un vecteur de steps[seed=1..50]
    qui est une empreinte du scénario de vent.

    À utiliser sur Codabench pour collecter les données du scénario test.
    Les steps sont imprimés en JSON dans stdout pour être récupérés.
    """

    def __init__(self):
        super().__init__()
        self.world_map   = None
        self._step       = 0
        # Persistance cross-seeds via variables de classe
        MyAgent._all_steps = getattr(MyAgent, '_all_steps', {})
        MyAgent._current_seed = getattr(MyAgent, '_current_seed', None)

    def reset(self):
        self.world_map = None
        self._step     = 0

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path): pass
    def load(self, path): pass

    def _parse(self, obs):
        px,py = float(obs[0]),float(obs[1])
        vx,vy = float(obs[2]),float(obs[3])
        wf = obs[6:6+128*128*2].reshape(128,128,2).astype(np.float32)
        if self.world_map is None:
            self.world_map = obs[6+128*128*2:].reshape(128,128).astype(np.float32)
        return px,py,vx,vy,wf

    def act(self, observation: np.ndarray) -> int:
        px,py,vx,vy,wf = self._parse(observation)
        self._step += 1
        return int(_windmaster_with_predicted_wind(px, py, vx, vy, wf, self.world_map))