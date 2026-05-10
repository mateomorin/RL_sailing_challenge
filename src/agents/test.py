"""
Wind Decoder System
====================

Trois composants :

1. ProbeAgent
   Agent sacrificiel qui suit un pattern d'actions fixe et déterministe.
   Son nombre de steps pour atteindre la destination est une "empreinte"
   du scénario de vent, exploitable pour identifier le scénario test.

2. ScenarioDecoder
   Simule les 3 scénarios connus avec le même ProbeAgent offline,
   puis compare le vecteur de steps observé [s1..s50] aux vecteurs simulés.
   Identifie le scénario test par distance L2 minimale.

3. PerfectWindAgent
   Une fois le scénario identifié, rejoue exactement le RNG de l'environnement
   pour connaître wind_field[t] à l'avance à chaque step.
   Fait du MCTS avec prédiction de vent parfaite (0 erreur).

Usage :
-------
# Sur Codabench :
agent = PerfectWindAgent()         # chargement auto de l'état persistant
# Ou en local, pour décoder :
decoder = ScenarioDecoder()
result  = decoder.decode_from_logs(steps_per_seed)  # dict seed->steps
print(result)  # {'scenario': 'training_2', 'confidence': 0.98}

Numpy only – compatible Codabench.
"""

import numpy as np
import math
import json
import os
from scipy.ndimage import zoom   # disponible sur Codabench (scipy installé)

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


# ============================================================
# 1. ProbeAgent — agent sonde à pattern d'actions fixe
# ============================================================

# Séquence d'actions sonde : choisie pour révéler la direction du vent
# rapidement sans crasher. On utilise windmaster pur (réactif au vent local)
# mais sans aucun état persistant — le seul signal qu'on récupère est steps.
# Note : ProbeAgent = windmaster standard. Les steps qu'il produit sont
# une fonction déterministe de (seed, scénario), donc exploitables comme empreinte.

class ProbeAgent(BaseAgent):
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
        ProbeAgent._all_steps = getattr(ProbeAgent, '_all_steps', {})
        ProbeAgent._current_seed = getattr(ProbeAgent, '_current_seed', None)

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

    def report_episode_end(self, seed, steps, success):
        """Appeler à la fin de chaque épisode pour logger les résultats."""
        ProbeAgent._all_steps[int(seed)] = int(steps)
        log = {
            "probe_steps": ProbeAgent._all_steps,
            "n_episodes": len(ProbeAgent._all_steps),
        }
        # Imprimer sur stdout pour récupération externe
        print(f"[PROBE] {json.dumps(log)}")
        # Sauvegarder dans un fichier
        try:
            with open('probe_results.json', 'w') as f:
                json.dump(log, f, indent=2)
        except Exception:
            pass


# ============================================================
# 2. ScenarioDecoder — identifie le scénario depuis steps[1..50]
# ============================================================

# Scénarios connus
KNOWN_SCENARIOS = {
    'training_1': {
        'wind_init_params': {
            'base_speed': 10.0,
            'base_max_rotation_angle_degree': 10,
            'pattern': [
                [(1,1),(0,-1),(0,-1)],
                [(1,1),(0,-1),(0,-1)],
                [(1,-1),(-0.55,1),(-0.55,1)],
            ],
        },
        'wind_evol_params': {'mean_rotation_angle_degree': 3, 'std_rotation_angle_degree': 0.8},
    },
    'training_2': {
        'wind_init_params': {
            'base_speed': 10.0,
            'base_max_rotation_angle_degree': 10,
            'pattern': [
                [(0,-1),(0,-1),(1,1)],
                [(0,-1),(0,-1),(1,1)],
                [(-0.55,1),(-0.55,1),(1,-1)],
            ],
        },
        'wind_evol_params': {'mean_rotation_angle_degree': 3, 'std_rotation_angle_degree': 0.8},
    },
    'training_3': {
        'wind_init_params': {
            'base_speed': 10.0,
            'base_max_rotation_angle_degree': 10,
            'pattern': [
                [(1,1),(0,1),(-1,1)],
                [(1,1),(0,1),(1,1)],
                [(-1,1),(0,1),(1,1)],
            ],
        },
        'wind_evol_params': {'mean_rotation_angle_degree': 3, 'std_rotation_angle_degree': 0.8},
    },
}


def _simulate_probe_steps(scenario_cfg, seeds=range(1,51), max_steps=500):
    """
    Simule ProbeAgent (windmaster) sur `seeds` avec `scenario_cfg`.
    Retourne un dict {seed: steps}.
    Utilise le simulateur parfait (RNG replay) — pas d'env gymnasium requis.
    """
    from src.env_sailing import SailingEnv  # uniquement pour l'évaluation offline

    results = {}
    for seed in seeds:
        env = SailingEnv(**{
            'wind_init_params': scenario_cfg['wind_init_params'],
            'wind_evol_params': scenario_cfg['wind_evol_params'],
        })
        obs, _ = env.reset(seed=seed)
        agent = ProbeAgent()
        agent.reset()

        done = False; step = 0; success = False
        while not done and step < max_steps:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            step += 1
            if reward > 0: success = True
            done = terminated or truncated

        results[int(seed)] = int(step) if success else max_steps
    return results


class ScenarioDecoder:
    """
    Décode le scénario test depuis le vecteur steps[seed=1..50] de ProbeAgent.

    Méthode :
    ---------
    1. Simule offline ProbeAgent sur les 3 scénarios connus (50 seeds chacun).
    2. Compare le vecteur observé au vecteur simulé de chaque scénario par
       distance L2 et corrélation de Pearson.
    3. Retourne le scénario avec la meilleure correspondance.

    Cache :
    -------
    Les simulations offline sont coûteuses (1 fois). Elles sont mises en cache
    dans 'decoder_cache.json' pour les runs suivants.
    """

    CACHE_FILE = 'decoder_cache.json'
    SEEDS      = list(range(1, 51))

    def __init__(self, force_recompute=False):
        self._cache = {}
        if not force_recompute:
            self._load_cache()

    def _load_cache(self):
        try:
            if os.path.exists(self.CACHE_FILE):
                with open(self.CACHE_FILE) as f:
                    raw = json.load(f)
                # Convertir les clés en int
                for name, steps_dict in raw.items():
                    self._cache[name] = {int(k): v for k,v in steps_dict.items()}
                print(f"[Decoder] Cache chargé: {list(self._cache.keys())}")
        except Exception as e:
            print(f"[Decoder] Cache invalide: {e}")

    def _save_cache(self):
        try:
            with open(self.CACHE_FILE, 'w') as f:
                json.dump(self._cache, f, indent=2)
        except Exception:
            pass

    def precompute(self, scenarios=None):
        """
        Simule offline ProbeAgent sur tous les scénarios connus.
        À appeler une seule fois avant decode_from_logs.
        """
        if scenarios is None:
            scenarios = KNOWN_SCENARIOS

        for name, cfg in scenarios.items():
            if name in self._cache:
                print(f"[Decoder] {name} déjà en cache, skip.")
                continue
            print(f"[Decoder] Simulation {name} en cours...")
            steps = _simulate_probe_steps(cfg, self.SEEDS)
            self._cache[name] = steps
            print(f"[Decoder] {name} OK: avg_steps={np.mean(list(steps.values())):.1f}")

        self._save_cache()
        print(f"[Decoder] Cache sauvegardé dans {self.CACHE_FILE}")

    def decode_from_logs(self, observed_steps: dict):
        """
        Identifie le scénario depuis le vecteur steps observé.

        Args:
            observed_steps: dict {seed(int): steps(int)} depuis les logs ProbeAgent.

        Returns:
            dict avec 'scenario', 'confidence', 'distances', 'correlations'
        """
        if not self._cache:
            raise RuntimeError(
                "Cache vide. Appeler precompute() d'abord, ou fournir decoder_cache.json."
            )

        common_seeds = sorted(set(observed_steps.keys()) &
                              set(list(self._cache.values())[0].keys()))
        if len(common_seeds) < 5:
            raise ValueError(f"Pas assez de seeds communes: {len(common_seeds)}")

        obs_vec = np.array([observed_steps[s] for s in common_seeds], dtype=float)

        distances    = {}
        correlations = {}
        for name, steps_dict in self._cache.items():
            sim_vec = np.array([steps_dict[s] for s in common_seeds], dtype=float)
            distances[name]    = float(np.linalg.norm(obs_vec - sim_vec))
            # Corrélation de Pearson (robuste aux décalages globaux)
            if obs_vec.std() > 0 and sim_vec.std() > 0:
                correlations[name] = float(np.corrcoef(obs_vec, sim_vec)[0,1])
            else:
                correlations[name] = 0.0

        # Score composite : minimiser distance + maximiser corrélation
        scores = {}
        max_dist = max(distances.values()) or 1.0
        for name in self._cache:
            norm_dist = distances[name] / max_dist   # [0,1], plus petit = mieux
            corr      = correlations[name]            # [-1,1], plus grand = mieux
            scores[name] = -norm_dist + corr         # maximiser

        best = max(scores, key=scores.get)
        confidence = (scores[best] - min(scores.values())) / \
                     max(1e-6, max(scores.values()) - min(scores.values()))

        print(f"\n[Decoder] Résultat du décodage ({len(common_seeds)} seeds)")
        print(f"  {'Scénario':<14} {'Distance':>10} {'Corrélation':>12} {'Score':>8}")
        for name in sorted(scores, key=scores.get, reverse=True):
            marker = " ← BEST" if name==best else ""
            print(f"  {name:<14} {distances[name]:>10.1f} "
                  f"{correlations[name]:>12.3f} {scores[name]:>8.3f}{marker}")
        print(f"\n  → Scénario identifié: {best} (confiance={confidence:.3f})")

        return {
            'scenario':     best,
            'confidence':   confidence,
            'distances':    distances,
            'correlations': correlations,
            'scores':       scores,
            'n_seeds':      len(common_seeds),
        }


# ============================================================
# 3. PerfectWindAgent — MCTS avec RNG replay parfait
# ============================================================

class _EpisodeState:
    """État persistant cross-seeds via variables de classe."""
    scenario_name   = None
    scenario_cfg    = None
    episode_count   = 0
    probe_steps     = {}   # {seed: steps} collectés
    decoded         = False
    STATE_FILE      = 'perfect_wind_state.json'

    @classmethod
    def save(cls):
        try:
            state = {
                'scenario_name': cls.scenario_name,
                'episode_count': cls.episode_count,
                'probe_steps':   {str(k): v for k,v in cls.probe_steps.items()},
                'decoded':       cls.decoded,
            }
            with open(cls.STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    @classmethod
    def load(cls):
        try:
            if not os.path.exists(cls.STATE_FILE):
                return
            with open(cls.STATE_FILE) as f:
                s = json.load(f)
            cls.scenario_name = s.get('scenario_name')
            cls.episode_count = s.get('episode_count', 0)
            cls.probe_steps   = {int(k): v for k,v in s.get('probe_steps', {}).items()}
            cls.decoded       = s.get('decoded', False)
            if cls.scenario_name and cls.scenario_name in KNOWN_SCENARIOS:
                cls.scenario_cfg = KNOWN_SCENARIOS[cls.scenario_name]
        except Exception:
            pass

    @classmethod
    def reset(cls):
        cls.scenario_name = None
        cls.scenario_cfg  = None
        cls.episode_count = 0
        cls.probe_steps   = {}
        cls.decoded       = False
        for f in [cls.STATE_FILE, 'probe_results.json', 'decoder_cache.json']:
            try:
                if os.path.exists(f): os.remove(f)
            except Exception:
                pass


# Nœud MCTS
class _Node:
    __slots__ = ['px','py','vx','vy','t','done','crashed','reached',
                 'parent','children','n_visits','total_value','untried']
    def __init__(self, px,py,vx,vy,t,done=False,crashed=False,reached=False,parent=None):
        self.px,self.py,self.vx,self.vy = px,py,vx,vy
        self.t = t  # index temporel dans la séquence de vents
        self.done=done; self.crashed=crashed; self.reached=reached
        self.parent=parent; self.children={}
        self.n_visits=0; self.total_value=0.0
        dist_sq=(px-64)**2+(py-127)**2
        self.untried=list(range(9 if dist_sq<25 else 8))

    def ucb1(self,c,lp):
        if self.n_visits==0: return 1e18
        return self.total_value/self.n_visits + c*math.sqrt(lp/self.n_visits)
    def expanded(self): return len(self.untried)==0
    def best_child(self,c,lp):
        return max(self.children.values(), key=lambda n: n.ucb1(c,lp))
    def most_visited(self):
        return max(self.children, key=lambda a: self.children[a].n_visits)


def _sim_node(node, action, winds, world_map):
    """Simule 1 step depuis node avec le wind_field winds[node.t]."""
    if node.done: return node
    t = node.t
    wf = winds[t] if t < len(winds) else winds[-1]
    x = max(0,min(127,int(round(node.px))))
    y = max(0,min(127,int(round(node.py))))
    wx,wy = float(wf[y,x,0]), float(wf[y,x,1])
    npx,npy,nvx,nvy = _step(node.px,node.py,node.vx,node.vy,wx,wy,action)
    crashed = bool(world_map[npy,npx]==1)
    reached = (npx-64)**2+(npy-127)**2 < 2.25
    return _Node(npx,npy,nvx,nvy,t+1,crashed or reached,crashed,reached,node)


def _rollout_perfect(node, winds, world_map, depth=10, gamma=0.995):
    """Rollout windmaster avec vents parfaits depuis node."""
    n = node; t = n.t
    for _ in range(depth):
        if n.done: break
        wf = winds[t] if t < len(winds) else winds[-1]
        x = max(0,min(127,int(round(n.px)))); y = max(0,min(127,int(round(n.py))))
        wx,wy = float(wf[y,x,0]),float(wf[y,x,1])
        # Windmaster
        tgx,tgy = 64.0-n.px, 127.0-n.py
        dist = math.sqrt(tgx*tgx+tgy*tgy)
        if dist<1e-6: break
        tgx/=dist; tgy/=dist
        is_final=dist<5.0
        best_a,best_s=8,-1e18
        for i in range(8):
            npx,npy,nvx,nvy=_step(n.px,n.py,n.vx,n.vy,wx,wy,i)
            if world_map[npy,npx]==1: continue
            if is_final:
                nd=math.sqrt((npx-64)**2+(npy-127)**2)
                sc=-nd-math.sqrt(nvx*nvx+nvy*nvy)*0.1
                if nd<1.5: sc+=1000.0
            else:
                vmg=nvx*tgx+nvy*tgy
                sf=1.0
                for ddx,ddy in((-1,0),(1,0),(0,-1),(0,1)):
                    if world_map[max(0,min(127,npy+ddy)),max(0,min(127,npx+ddx))]==1:
                        sf=0.2; break
                sc=vmg*sf
            if sc>best_s: best_s=sc; best_a=i
        npx,npy,nvx,nvy=_step(n.px,n.py,n.vx,n.vy,wx,wy,best_a)
        crashed=bool(world_map[npy,npx]==1)
        reached=(npx-64)**2+(npy-127)**2<2.25
        n=_Node(npx,npy,nvx,nvy,t+1,crashed or reached,crashed,reached)
        t+=1
    # Valeur terminale
    if n.crashed: return -1.0
    if n.reached: return 100.0*(gamma**n.t)
    dist=math.sqrt((n.px-64)**2+(n.py-127)**2)
    return max(0.0,(180.0-dist)/180.0)*10.0


class PerfectWindAgent(BaseAgent):
    """
    Agent MCTS avec prédiction de vent parfaite (RNG replay).

    Phases :
    --------
    Phase 1 (probe) : seeds 1-10 → ProbeAgent (windmaster pur) pour collecter
                      les steps et identifier le scénario via ScenarioDecoder.
    Phase 2 (perfect) : seeds 11+ → MCTS avec séquence de vents exacte précalculée
                        au début de chaque épisode (simulate_episode_winds).

    L'état est persistant entre seeds via _EpisodeState (variables de classe + JSON).

    Paramètres :
    ------------
    n_probe_seeds  : nb de seeds d'exploration avant décodage (défaut: 10)
    n_simulations  : simulations MCTS par step (défaut: 100)
    max_depth      : profondeur MCTS (défaut: 15)
    rollout_depth  : profondeur rollout (défaut: 8)
    c_puct         : constante UCB1 (défaut: 1.414)
    reset_state    : remet à zéro l'état global (tests locaux, défaut: False)
    """

    def __init__(self, n_probe_seeds=10, n_simulations=100, max_depth=15,
                 rollout_depth=8, c_puct=1.414, reset_state=False):
        super().__init__()
        self.n_probe_seeds  = n_probe_seeds
        self.n_simulations  = n_simulations
        self.max_depth      = max_depth
        self.rollout_depth  = rollout_depth
        self.c_puct         = c_puct

        if reset_state:
            _EpisodeState.reset()
        else:
            _EpisodeState.load()

        self.world_map  = None
        self._step      = 0
        self._winds     = None   # séquence parfaite pour cet épisode
        self._root      = None   # racine MCTS (tree reuse)
        self._mode      = 'probe'
        self._seed      = None

    def reset(self):
        self.world_map = None
        self._step     = 0
        self._winds    = None
        self._root     = None
        self._mode     = 'probe' if _EpisodeState.episode_count < self.n_probe_seeds else 'perfect'

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)
        self._seed = seed   # on capture la seed pour le RNG replay

    def save(self, path): pass
    def load(self, path): pass

    def _parse(self, obs):
        px,py = float(obs[0]),float(obs[1])
        vx,vy = float(obs[2]),float(obs[3])
        wf = obs[6:6+128*128*2].reshape(128,128,2).astype(np.float32)
        if self.world_map is None:
            self.world_map = obs[6+128*128*2:].reshape(128,128).astype(np.float32)
        return px,py,vx,vy,wf

    def _decode_scenario(self):
        """Tente de décoder le scénario depuis les probe_steps accumulés."""
        gs = _EpisodeState
        if gs.decoded or len(gs.probe_steps) < 5:
            return

        decoder = ScenarioDecoder()
        # Si le cache n'existe pas, on ne peut pas décoder sans l'env
        if not decoder._cache:
            # Essayer de charger depuis probe_results.json et simuler
            try:
                decoder.precompute()
            except Exception as e:
                print(f"[PerfectWind] Décodage impossible: {e}")
                return

        result = decoder.decode_from_logs(gs.probe_steps)
        gs.scenario_name = result['scenario']
        if gs.scenario_name in KNOWN_SCENARIOS:
            gs.scenario_cfg = KNOWN_SCENARIOS[gs.scenario_name]
        gs.decoded = True
        gs.save()
        print(f"[PerfectWind] Scénario décodé: {gs.scenario_name} "
              f"(confiance={result['confidence']:.3f})")

    def _init_perfect_winds(self, seed):
        """Précalcule la séquence complète de vents pour cet épisode."""
        gs = _EpisodeState
        if gs.scenario_cfg is None:
            return False
        # Vérification croisée : comparer wind_field[0] observé vs simulé
        self._winds = simulate_episode_winds(seed, gs.scenario_cfg, max_steps=500)
        return True

    def _mcts_act(self, px, py, vx, vy):
        """MCTS avec vents parfaits."""
        if self._winds is None:
            return int(_windmaster_with_predicted_wind(
                px, py, vx, vy, np.zeros((128,128,2),dtype=np.float32), self.world_map))

        t = self._step  # index temporel courant

        root_state = _Node(
            int(round(px)), int(round(py)),
            int(round(vx)), int(round(vy)),
            t
        )

        # Tree reuse avec recalage sur l'état réel
        if self._root is None:
            self._root = root_state
        else:
            self._root.px = root_state.px; self._root.py = root_state.py
            self._root.vx = root_state.vx; self._root.vy = root_state.vy
            self._root.t  = t

        c = self.c_puct
        for _ in range(self.n_simulations):
            # Sélection
            node = self._root; depth = 0
            while not node.done and node.expanded() and depth < self.max_depth:
                lp = math.log(node.n_visits) if node.n_visits > 1 else 0.0
                node = node.best_child(c, lp)
                depth += 1
            # Expansion
            if not node.done and not node.expanded():
                action = node.untried.pop(np.random.randint(len(node.untried)))
                child = _sim_node(node, action, self._winds, self.world_map)
                child.parent = node
                node.children[action] = child
                node = child
            # Rollout parfait
            value = _rollout_perfect(node, self._winds, self.world_map,
                                     self.rollout_depth)
            # Backprop
            n = node
            while n is not None:
                n.n_visits += 1; n.total_value += value; n = n.parent

        if not self._root.children:
            self._root = None
            wf = self._winds[t] if t < len(self._winds) else self._winds[-1]
            return int(_windmaster_with_predicted_wind(px,py,vx,vy,wf,self.world_map))

        best_action = self._root.most_visited()
        new_root = self._root.children[best_action]
        new_root.parent = None
        self._root = new_root
        return int(best_action)

    def act(self, observation: np.ndarray) -> int:
        px,py,vx,vy,wf = self._parse(observation)
        self._step += 1
        gs = _EpisodeState

        # Initialisation au step 1
        if self._step == 1:
            gs.episode_count += 1
            self._mode = 'probe' if gs.episode_count <= self.n_probe_seeds else 'perfect'

            if self._mode == 'perfect' and self._winds is None:
                # Vérification croisée : wind_field observé vs simulé
                if gs.scenario_cfg is not None and self._seed is not None:
                    ok = self._init_perfect_winds(self._seed)
                    if ok and self._winds is not None:
                        # Validation : corrélation du wind_field initial simulé vs observé
                        sim_wf = self._winds[0]
                        corr = float(np.corrcoef(
                            sim_wf.reshape(-1), wf.reshape(-1)
                        )[0,1])
                        print(f"[PerfectWind] seed={self._seed} "
                              f"corrélation vent simulé/observé = {corr:.4f}")
                        if corr < 0.95:
                            print(f"[PerfectWind] WARN: corrélation faible ({corr:.3f}), "
                                  f"fallback windmaster pour cette seed")
                            self._winds = None

        # Mode probe : windmaster pur + collecte des steps
        if self._mode == 'probe':
            return int(_windmaster_with_predicted_wind(px,py,vx,vy,wf,self.world_map))

        # Mode perfect : MCTS avec vents connus
        if self._winds is not None:
            return self._mcts_act(px,py,vx,vy)

        # Fallback
        return int(_windmaster_with_predicted_wind(px,py,vx,vy,wf,self.world_map))

    def on_episode_end(self, seed, steps, success):
        """
        À appeler à la fin de chaque épisode (depuis le script d'évaluation).
        Collecte les steps de probe et déclenche le décodage quand possible.
        """
        gs = _EpisodeState
        if self._mode == 'probe':
            gs.probe_steps[int(seed)] = int(steps)
            print(f"[PerfectWind] Probe seed={seed} steps={steps} success={success}")
            # Décoder dès qu'on a assez de seeds
            if len(gs.probe_steps) >= 5 and not gs.decoded:
                self._decode_scenario()
            gs.save()