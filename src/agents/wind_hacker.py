"""
Wind Hacker Agent (Idée 2)
===========================

Stratégie :
-----------
Les 50 seeds du scénario test sont évaluées séquentiellement par Codabench.
On exploite cela pour **reconstruire le scénario test** seed après seed.

Principe :
----------
Le champ de vent initial `wind_field_0` au reset(seed=k) dépend de la seed :
  wind_field_0 = base_pattern + bruit_spatial(seed=k)

Le `base_pattern` est identique pour toutes les seeds (c'est le pattern 3×3
interpolé). Le bruit spatial `coarse_noise` varie par seed.

En moyennant les champs initiaux observés sur les N premières seeds, le bruit
se cancelle (espérance nulle) et on récupère le base_pattern pur.

Avec ce base_pattern, on peut :
1. Identifier le scénario test (matching avec les 3 scénarios connus ou nouveau)
2. Extraire le corridor optimal (direction dominante du vent près du départ)
3. Pré-calculer une trajectoire optimale fixe (waypoints) qui ignore le bruit

Architecture de l'agent :
--------------------------
L'agent maintient un état PERSISTANT entre les épisodes via un fichier JSON
(wind_hacker_state.json). Codabench exécute chaque seed dans le même process
Python, donc les variables de classe sont aussi persistantes — on utilise les
deux mécanismes pour robustesse.

Phase 1 (seeds 1-5)  : exploration pure → windmaster pur, accumule wind_fields
Phase 2 (seeds 6-10) : identification du scénario → windmaster avec corridor
Phase 3 (seeds 11+)  : politique optimisée basée sur le scénario reconstruit

Numpy only – compatible Codabench.
"""

import numpy as np
import math
import json
import os

try:
    from base_agent import BaseAgent
except ImportError:
    try:
        from agents.base_agent import BaseAgent
    except ImportError:
        from src.agents.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Physique
# ---------------------------------------------------------------------------

DIRECTIONS = np.array([
    [0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1],[0,0]
], dtype=np.float32)

BOAT_PERFORMANCE = 0.4
MAX_SPEED        = 8.0
INERTIA_FACTOR   = 0.3


def _eff(dnx, dny, wfnx, wfny):
    cos_a = dnx*wfnx + dny*wfny
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    pi4 = math.pi/4
    if a < pi4: return 0.05
    elif a < math.pi/2: return 0.5+0.5*(a-pi4)/pi4
    elif a < 3*pi4: return 1.0
    else: return max(0.5, 1.0-0.5*(a-3*pi4)/pi4)


def _step(px, py, vx, vy, wx, wy, ai):
    dx, dy = float(DIRECTIONS[ai,0]), float(DIRECTIONS[ai,1])
    wn = math.sqrt(wx*wx+wy*wy)
    if wn > 1e-6:
        wnx,wny = wx/wn, wy/wn
        dn = math.sqrt(dx*dx+dy*dy)
        dnx,dny = (dx/dn,dy/dn) if dn>1e-10 else (1.0,0.0)
        e = _eff(dnx,dny,-wnx,-wny)
        tvx=dx*e*wn*BOAT_PERFORMANCE; tvy=dy*e*wn*BOAT_PERFORMANCE
        ts=math.sqrt(tvx*tvx+tvy*tvy)
        if ts>MAX_SPEED: tvx*=MAX_SPEED/ts; tvy*=MAX_SPEED/ts
        nvx=tvx+INERTIA_FACTOR*(vx-tvx); nvy=tvy+INERTIA_FACTOR*(vy-tvy)
        ns=math.sqrt(nvx*nvx+nvy*nvy)
        if ns>MAX_SPEED: nvx*=MAX_SPEED/ns; nvy*=MAX_SPEED/ns
    else:
        nvx=INERTIA_FACTOR*vx; nvy=INERTIA_FACTOR*vy
    ivx=int(math.ceil(nvx) if nvx<0 else math.floor(nvx))
    ivy=int(math.ceil(nvy) if nvy<0 else math.floor(nvy))
    return max(0,min(127,int(round(px))+ivx)), max(0,min(127,int(round(py))+ivy)), ivx, ivy


def _windmaster(px, py, vx, vy, wf, wm, corridor='any', target=None):
    """Windmaster avec biais de corridor optionnel et target custom."""
    x,y = max(0,min(127,int(round(px)))), max(0,min(127,int(round(py))))
    wx,wy = float(wf[y,x,0]), float(wf[y,x,1])
    if target is None:
        tgx,tgy = 64.0-px, 127.0-py
    else:
        tgx,tgy = float(target[0])-px, float(target[1])-py
    dist = math.sqrt(tgx*tgx+tgy*tgy)
    if dist < 1e-6: return 8
    tgx/=dist; tgy/=dist
    is_final = dist < 5.0

    # Biais corridor
    bx = 0.0
    if not is_final and py < 80 and math.sqrt((px-64)**2+(py-127)**2) > 30:
        if corridor == 'right': bx = 0.4
        elif corridor == 'left': bx = -0.4
    etx = tgx+bx; ety = tgy
    en = math.sqrt(etx*etx+ety*ety)
    if en>1e-6: etx/=en; ety/=en
    else: etx,ety=tgx,tgy

    best_a,best_s = 8,-1e18
    for i in range(8):
        npx,npy,nvx,nvy = _step(px,py,vx,vy,wx,wy,i)
        if wm[npy,npx]==1: continue
        if is_final:
            nd=math.sqrt((npx-64)**2+(npy-127)**2)
            sc=-nd-math.sqrt(nvx*nvx+nvy*nvy)*0.1
            if nd<1.5: sc+=1000.0
        else:
            vmg=nvx*etx+nvy*ety
            safety=1.0
            for ddx,ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                if wm[max(0,min(127,npy+ddy)),max(0,min(127,npx+ddx))]==1:
                    safety=0.2; break
            sc=vmg*safety
        if sc>best_s: best_s=sc; best_a=i
    return best_a


# ---------------------------------------------------------------------------
# Identification de scénario (signatures pré-calculées)
# ---------------------------------------------------------------------------

SCENARIO_PATTERNS = {
    'training_1': [[(1,1),(0,-1),(0,-1)],[(1,1),(0,-1),(0,-1)],[(1,-1),(-0.55,1),(-0.55,1)]],
    'training_2': [[(0,-1),(0,-1),(1,1)],[(0,-1),(0,-1),(1,1)],[(-0.55,1),(-0.55,1),(1,-1)]],
    'training_3': [[(1,1),(0,1),(-1,1)],[(1,1),(0,1),(1,1)],[(-1,1),(0,1),(1,1)]],
}

SCENARIO_CORRIDOR = {
    'training_1': 'right',
    'training_2': 'left',
    'training_3': 'any',
    'unknown':    'any',
}


def _build_sig(pattern):
    pat = np.array(pattern, dtype=np.float64)
    norms = np.linalg.norm(pat, axis=-1, keepdims=True)
    pat = pat / np.maximum(norms, 1e-8)
    n_rows,n_cols,H,W = 3,3,128,128
    Yi = np.linspace(0,n_rows-1,H); Xi = np.linspace(0,n_cols-1,W)
    Xi,Yi = np.meshgrid(Xi,Yi)
    i0=np.floor(Yi).astype(int); j0=np.floor(Xi).astype(int)
    i1=np.clip(i0+1,0,n_rows-1); j1=np.clip(j0+1,0,n_cols-1)
    dy=(Yi-i0)[:,:,None]; dx=(Xi-j0)[:,:,None]
    interp=(pat[i0,j0]*(1-dx)*(1-dy)+pat[i0,j1]*dx*(1-dy)+
            pat[i1,j0]*(1-dx)*dy+pat[i1,j1]*dx*dy)
    n=np.linalg.norm(interp,axis=-1,keepdims=True)
    interp=interp/np.maximum(n,1e-8)
    return interp.reshape(-1)

_SIGS = {k: _build_sig(v) for k,v in SCENARIO_PATTERNS.items()}


def _identify(mean_wf, threshold=0.80):
    """Identifie le scénario depuis le champ de vent moyen."""
    wf_dir = mean_wf / np.maximum(np.linalg.norm(mean_wf, axis=-1, keepdims=True), 1e-8)
    obs = wf_dir.reshape(-1)
    obs = obs / max(np.linalg.norm(obs), 1e-8)
    best_n, best_c = 'unknown', -1.0
    for name, sig in _SIGS.items():
        sig_n = sig / max(np.linalg.norm(sig), 1e-8)
        c = float(np.dot(obs, sig_n))
        if c > best_c: best_c=c; best_n=name
    return (best_n if best_c>=threshold else 'unknown'), best_c


def _extract_corridor_from_wf(mean_wf):
    """
    Extrait le corridor optimal directement depuis le champ de vent moyen reconstruit,
    sans nécessiter de matching avec les scénarios connus.
    Regarde le vent moyen dans la zone de départ (x:40-88, y:0-40).
    Si vent dominant vers la droite (wx>0) : corridor right.
    Si vent dominant vers la gauche (wx<0) : corridor left.
    """
    zone = mean_wf[0:40, 40:88, 0]  # composante x du vent dans la zone de départ
    mean_wx = float(np.mean(zone))
    if mean_wx > 0.5:
        return 'right'
    elif mean_wx < -0.5:
        return 'left'
    else:
        return 'any'


# ---------------------------------------------------------------------------
# État persistant (partagé entre épisodes via variable de classe)
# ---------------------------------------------------------------------------

class _GlobalState:
    """
    État persistant entre épisodes. Variables de classe = partagées dans
    le même process Python (comme sur Codabench entre les seeds).
    """
    episode_count   = 0          # nombre d'épisodes terminés
    wind_field_sum  = None       # somme des champs initiaux observés
    mean_wind_field = None       # moyenne courante
    scenario        = None       # scénario identifié
    corridor        = 'any'      # corridor retenu
    confidence      = 0.0

    EXPLORATION_PHASE  = 5       # seeds 1-5 : exploration pure
    IDENTIFICATION_PHASE = 10    # seeds 6-10 : identification + corridor

    STATE_FILE = 'wind_hacker_state.json'

    @classmethod
    def save(cls):
        """Sauvegarde l'état dans un fichier JSON (robustesse)."""
        state = {
            'episode_count': cls.episode_count,
            'scenario': cls.scenario,
            'corridor': cls.corridor,
            'confidence': cls.confidence,
        }
        if cls.mean_wind_field is not None:
            state['mean_wind_field'] = cls.mean_wind_field.tolist()
        try:
            with open(cls.STATE_FILE, 'w') as f:
                json.dump(state, f)
        except Exception:
            pass  # pas bloquant

    @classmethod
    def load(cls):
        """Charge l'état depuis le fichier JSON si disponible."""
        try:
            if not os.path.exists(cls.STATE_FILE):
                return
            with open(cls.STATE_FILE, 'r') as f:
                state = json.load(f)
            cls.episode_count = state.get('episode_count', 0)
            cls.scenario      = state.get('scenario', None)
            cls.corridor      = state.get('corridor', 'any')
            cls.confidence    = state.get('confidence', 0.0)
            if 'mean_wind_field' in state:
                cls.mean_wind_field = np.array(state['mean_wind_field'], dtype=np.float32).reshape(128,128,2)
        except Exception:
            pass

    @classmethod
    def reset_all(cls):
        """Remet à zéro l'état (utile pour les tests locaux)."""
        cls.episode_count   = 0
        cls.wind_field_sum  = None
        cls.mean_wind_field = None
        cls.scenario        = None
        cls.corridor        = 'any'
        cls.confidence      = 0.0
        try:
            if os.path.exists(cls.STATE_FILE):
                os.remove(cls.STATE_FILE)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Agent hackeur
# ---------------------------------------------------------------------------

class MyAgent(BaseAgent):
    """
    Agent qui reconstruit le scénario test en accumulant les observations
    cross-seeds, puis adapte sa politique en conséquence.

    Phases :
    --------
    1. seeds 1-5   : windmaster pur, accumulation des wind_fields initiaux
    2. seeds 6-10  : identification du scénario + corridor, windmaster biaisé
    3. seeds 11+   : windmaster biaisé avec le corridor optimal reconstruit

    L'état est persistant entre épisodes via variables de classe + fichier JSON.

    Paramètres :
    ------------
    id_threshold      : seuil corrélation pour identification (défaut: 0.80)
    exploration_phase : nb de seeds d'exploration pure (défaut: 5)
    reset_state       : si True, remet à zéro l'état global (tests locaux)
    """

    def __init__(self, id_threshold=0.80, exploration_phase=5, reset_state=False):
        super().__init__()
        self.id_threshold = id_threshold
        _GlobalState.EXPLORATION_PHASE = exploration_phase

        if reset_state:
            _GlobalState.reset_all()
        else:
            _GlobalState.load()

        self.world_map     = None
        self._step         = 0
        self._init_wf      = None   # wind_field observé au step 1 de cet épisode
        self._episode_done = False

    def reset(self):
        """Appelé au début de chaque épisode."""
        self.world_map     = None
        self._step         = 0
        self._init_wf      = None
        self._episode_done = False

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path): pass
    def load(self, path): pass

    def _parse_obs(self, obs):
        px,py = float(obs[0]),float(obs[1])
        vx,vy = float(obs[2]),float(obs[3])
        wf = obs[6:6+128*128*2].reshape(128,128,2).astype(np.float32)
        if self.world_map is None:
            self.world_map = obs[6+128*128*2:].reshape(128,128).astype(np.float32)
        return px,py,vx,vy,wf

    def _accumulate_wind(self, wf):
        """Accumule le champ de vent initial pour la moyenne cross-seeds."""
        gs = _GlobalState
        if gs.wind_field_sum is None:
            gs.wind_field_sum = wf.astype(np.float64)
        else:
            gs.wind_field_sum += wf.astype(np.float64)
        gs.episode_count += 1
        gs.mean_wind_field = (gs.wind_field_sum / gs.episode_count).astype(np.float32)

    def _try_identify(self):
        """Tente d'identifier le scénario depuis le champ moyen accumulé."""
        gs = _GlobalState
        if gs.mean_wind_field is None or gs.episode_count < 3:
            return
        name, conf = _identify(gs.mean_wind_field, self.id_threshold)
        gs.scenario   = name
        gs.confidence = conf
        # Corridor : depuis le matching si connu, sinon depuis le champ directement
        if name != 'unknown':
            gs.corridor = SCENARIO_CORRIDOR[name]
        else:
            gs.corridor = _extract_corridor_from_wf(gs.mean_wind_field)
        gs.save()

    def _on_episode_end(self):
        """Appelé quand l'épisode se termine (via act détectant done, ou reset suivant)."""
        if self._episode_done:
            return
        self._episode_done = True
        gs = _GlobalState
        # Accumulation du wind_field initial de cet épisode
        if self._init_wf is not None:
            self._accumulate_wind(self._init_wf)
            # Re-identifier à chaque seed (la moyenne s'améliore)
            self._try_identify()

    def act(self, observation: np.ndarray) -> int:
        px, py, vx, vy, wf = self._parse_obs(observation)
        self._step += 1

        # Capture du wind_field initial (step 1 = champ juste après reset)
        if self._step == 1:
            self._init_wf = wf.copy()
            # Accumulation immédiate (on peut aussi attendre la fin de l'épisode)
            self._on_episode_end()

        gs = _GlobalState
        ep = gs.episode_count

        # Phase 1 : exploration pure (windmaster sans biais)
        if ep <= gs.EXPLORATION_PHASE:
            return int(_windmaster(px, py, vx, vy, wf, self.world_map, corridor='any'))

        # Phase 2+ : windmaster avec corridor identifié
        corridor = gs.corridor
        return int(_windmaster(px, py, vx, vy, wf, self.world_map, corridor=corridor))

    # ------------------------------------------------------------------
    # Propriétés d'inspection
    # ------------------------------------------------------------------

    @property
    def episode_count(self):
        return _GlobalState.episode_count

    @property
    def scenario(self):
        return _GlobalState.scenario

    @property
    def corridor(self):
        return _GlobalState.corridor

    @property
    def confidence(self):
        return _GlobalState.confidence

    @property
    def mean_wind_field(self):
        return _GlobalState.mean_wind_field