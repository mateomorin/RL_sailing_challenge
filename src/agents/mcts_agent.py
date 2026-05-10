"""
MCTS Sailing Agent v3
=====================

Corrections vs v2 :

  v2 faisait du tree reuse en recalant seulement l'état de la RACINE sur
  l'observation réelle, mais laissait les états des nœuds ENFANTS intacts.
  Ces enfants avaient été calculés depuis l'ancienne racine simulée, avec un
  wind_angle_offset cumulé décalé du vrai vent. UCB1 guidait donc vers des
  branches dont les états position/collision étaient faux → crash massif.

  v3 : lors du tree reuse, on conserve UNIQUEMENT les statistiques UCB1
  (n_visits, total_value) des enfants. Leurs états (SimState) sont marqués
  invalides. La phase d'expansion recalcule l'état d'un enfant à la volée
  depuis l'état réel de son parent, en utilisant le wind_field fraîchement
  observé. Les états redeviennent valides et cohérents avec la réalité.

  L'arbre conserve ainsi sa mémoire statistique (quelles actions ont tendance
  à bien marcher) sans être pollué par des états simulés périmés.

  Autres améliorations conservées de v2 :
  - wind_angle_offset par nœud (pas de copie du wind_field 128×128×2)
  - scalaires Python dans la boucle physique (pas d'allocation numpy interne)
  - exclusion de "stay" loin du goal

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
# Constantes
# ---------------------------------------------------------------------------

DIRECTIONS = np.array([
    [0, 1],   # 0: N
    [1, 1],   # 1: NE
    [1, 0],   # 2: E
    [1, -1],  # 3: SE
    [0, -1],  # 4: S
    [-1, -1], # 5: SW
    [-1, 0],  # 6: W
    [-1, 1],  # 7: NW
    [0, 0],   # 8: Stay
], dtype=np.float32)

BOAT_PERFORMANCE = 0.4
MAX_SPEED        = 8.0
INERTIA_FACTOR   = 0.3


# ---------------------------------------------------------------------------
# Physique (scalaires Python, zéro allocation dans la boucle interne)
# ---------------------------------------------------------------------------

def _sailing_efficiency(dir_nx, dir_ny, wind_from_nx, wind_from_ny):
    cos_a = dir_nx * wind_from_nx + dir_ny * wind_from_ny
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    pi4 = math.pi / 4
    if a < pi4:
        return 0.05
    elif a < math.pi / 2:
        return 0.5 + 0.5 * (a - pi4) / pi4
    elif a < 3 * pi4:
        return 1.0
    else:
        return max(0.5, 1.0 - 0.5 * (a - 3 * pi4) / pi4)


def _get_wind_at(wind_field, px, py, angle_offset_rad):
    """Vent au pixel (px,py) avec offset angulaire. Rotation sur 1 pixel uniquement."""
    x = max(0, min(127, int(round(px))))
    y = max(0, min(127, int(round(py))))
    wx = float(wind_field[y, x, 0])
    wy = float(wind_field[y, x, 1])
    if angle_offset_rad != 0.0:
        c, s = math.cos(angle_offset_rad), math.sin(angle_offset_rad)
        wx, wy = wx * c - wy * s, wx * s + wy * c
    return wx, wy


def _step_physics(px, py, vx, vy, wx, wy, action_idx):
    """
    Un step complet de physique en scalaires.
    Retourne (new_px, new_py, new_vx, new_vy) – tous entiers.
    """
    dx = float(DIRECTIONS[action_idx, 0])
    dy = float(DIRECTIONS[action_idx, 1])
    wn = math.sqrt(wx * wx + wy * wy)

    if wn > 1e-6:
        wnx, wny = wx / wn, wy / wn
        dn = math.sqrt(dx * dx + dy * dy)
        if dn < 1e-10:
            dnx, dny = 1.0, 0.0
        else:
            dnx, dny = dx / dn, dy / dn

        eff = _sailing_efficiency(dnx, dny, -wnx, -wny)

        tvx = dx * eff * wn * BOAT_PERFORMANCE
        tvy = dy * eff * wn * BOAT_PERFORMANCE
        ts = math.sqrt(tvx * tvx + tvy * tvy)
        if ts > MAX_SPEED:
            tvx *= MAX_SPEED / ts
            tvy *= MAX_SPEED / ts

        nvx = tvx + INERTIA_FACTOR * (vx - tvx)
        nvy = tvy + INERTIA_FACTOR * (vy - tvy)
        ns = math.sqrt(nvx * nvx + nvy * nvy)
        if ns > MAX_SPEED:
            nvx *= MAX_SPEED / ns
            nvy *= MAX_SPEED / ns
    else:
        nvx = INERTIA_FACTOR * vx
        nvy = INERTIA_FACTOR * vy

    ivx = int(math.ceil(nvx) if nvx < 0 else math.floor(nvx))
    ivy = int(math.ceil(nvy) if nvy < 0 else math.floor(nvy))
    npx = max(0, min(127, int(round(px)) + ivx))
    npy = max(0, min(127, int(round(py)) + ivy))
    return npx, npy, ivx, ivy


# ---------------------------------------------------------------------------
# SimState
# ---------------------------------------------------------------------------

class SimState:
    """
    État simulé léger.
    wind_angle_offset : rotation cumulée par rapport au wind_field observé.
    """
    __slots__ = ['px', 'py', 'vx', 'vy', 'wind_angle_offset',
                 'step', 'done', 'crashed', 'reached']

    def __init__(self, px, py, vx, vy, wind_angle_offset=0.0,
                 step=0, done=False, crashed=False, reached=False):
        self.px = px; self.py = py
        self.vx = vx; self.vy = vy
        self.wind_angle_offset = wind_angle_offset
        self.step = step
        self.done = done; self.crashed = crashed; self.reached = reached


def _apply_sim_step(state, action_idx, world_map, wind_field, mean_rot_rad):
    """Retourne un nouveau SimState après l'action, ou state si terminal."""
    if state.done:
        return state
    wx, wy = _get_wind_at(wind_field, state.px, state.py, state.wind_angle_offset)
    npx, npy, nvx, nvy = _step_physics(
        state.px, state.py, state.vx, state.vy, wx, wy, action_idx
    )
    crashed = bool(world_map[npy, npx] == 1)
    reached = (npx - 64) ** 2 + (npy - 127) ** 2 < 2.25
    return SimState(
        px=npx, py=npy, vx=nvx, vy=nvy,
        wind_angle_offset=state.wind_angle_offset + mean_rot_rad,
        step=state.step + 1,
        done=crashed or reached,
        crashed=crashed, reached=reached,
    )


# ---------------------------------------------------------------------------
# Politique de rollout (windmaster allégé)
# ---------------------------------------------------------------------------

def _rollout_action(state, world_map, wind_field):
    wx, wy = _get_wind_at(wind_field, state.px, state.py, state.wind_angle_offset)
    tgx = 64.0 - state.px
    tgy = 127.0 - state.py
    dist = math.sqrt(tgx * tgx + tgy * tgy)
    if dist < 1e-6:
        return 8
    tgx /= dist; tgy /= dist
    is_final = dist < 5.0
    best_a, best_s = 8, -1e18
    for i in range(8):
        npx, npy, nvx, nvy = _step_physics(
            state.px, state.py, state.vx, state.vy, wx, wy, i
        )
        if world_map[npy, npx] == 1:
            continue
        if is_final:
            nd = math.sqrt((npx - 64) ** 2 + (npy - 127) ** 2)
            sc = -nd - math.sqrt(nvx * nvx + nvy * nvy) * 0.1
            if nd < 1.5:
                sc += 1000.0
        else:
            vmg = nvx * tgx + nvy * tgy
            safety = 1.0
            for ddx, ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                if world_map[max(0,min(127,npy+ddy)), max(0,min(127,npx+ddx))] == 1:
                    safety = 0.2; break
            sc = vmg * safety
        if sc > best_s:
            best_s = sc; best_a = i
    return best_a


def _terminal_value(state, gamma):
    if state.crashed:
        return -1.0
    if state.reached:
        return 100.0 * (gamma ** state.step)
    dist = math.sqrt((state.px - 64) ** 2 + (state.py - 127) ** 2)
    return max(0.0, (180.0 - dist) / 180.0) * 10.0


# ---------------------------------------------------------------------------
# Nœud MCTS
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    Nœud MCTS.

    state       : SimState courant — peut être None si le nœud a été hérité
                  par tree reuse et que son état n'a pas encore été recalculé.
    state_valid : True si state a été calculé depuis le vrai wind_field observé
                  à ce step. False si hérité d'un step précédent (état périmé).
                  Quand False, l'état est recalculé lors de la première expansion.
    action      : action qui a mené à ce nœud depuis son parent.
    """
    __slots__ = ['state', 'state_valid', 'action', 'parent', 'children',
                 'n_visits', 'total_value', 'untried_actions']

    def __init__(self, state, parent=None, action=None, state_valid=True):
        self.state       = state
        self.state_valid = state_valid
        self.action      = action
        self.parent      = parent
        self.children    = {}
        self.n_visits    = 0
        self.total_value = 0.0
        # "stay" inutile loin du goal
        if state is not None:
            dist_sq = (state.px - 64)**2 + (state.py - 127)**2
            self.untried_actions = list(range(9 if dist_sq < 25 else 8))
        else:
            self.untried_actions = list(range(8))

    def ucb1(self, c, log_parent):
        if self.n_visits == 0:
            return 1e18
        return self.total_value / self.n_visits + c * math.sqrt(log_parent / self.n_visits)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c, log_parent):
        return max(self.children.values(), key=lambda ch: ch.ucb1(c, log_parent))

    def most_visited_action(self):
        return max(self.children, key=lambda a: self.children[a].n_visits)


# ---------------------------------------------------------------------------
# Agent MCTS v3
# ---------------------------------------------------------------------------

class MyAgent(BaseAgent):
    """
    Agent MCTS pour le Sailing Challenge — v3.

    Tree reuse avec invalidation des états hérités :
    - Les statistiques UCB1 (n_visits, total_value) sont conservées entre steps.
    - Les états SimState hérités sont marqués invalides (state_valid=False).
    - L'état est recalculé depuis le parent lors de la première expansion,
      en utilisant le wind_field réel observé à ce step.
    → Plus de crash dû à des états simulés périmés.

    Paramètres :
    ------------
    n_simulations  : simulations MCTS par step (défaut: 100)
    max_depth      : profondeur max d'expansion (défaut: 15)
    rollout_depth  : steps de rollout windmaster (défaut: 5)
    c_puct         : constante UCB1 (défaut: 1.414)
    mean_rotation  : rotation moyenne du vent par step, degrés (défaut: 3.0)
    gamma          : discount factor (défaut: 0.995)
    """

    def __init__(self,
                 n_simulations=100,
                 max_depth=15,
                 rollout_depth=5,
                 c_puct=1.414,
                 mean_rotation=3.0,
                 gamma=0.995):
        super().__init__()
        self.n_simulations    = n_simulations
        self.max_depth        = max_depth
        self.rollout_depth    = rollout_depth
        self.c_puct           = c_puct
        self.mean_rot_rad     = math.radians(mean_rotation)
        self.gamma            = gamma
        self.world_map        = None
        self._step_count      = 0
        self._root            = None

    def reset(self):
        self.world_map   = None
        self._step_count = 0
        self._root       = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path): pass
    def load(self, path): pass

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_obs(self, obs):
        px, py = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        wf = obs[6:6 + 128*128*2].reshape(128, 128, 2).astype(np.float32)
        if self.world_map is None:
            self.world_map = obs[6 + 128*128*2:].reshape(128, 128).astype(np.float32)
        return px, py, vx, vy, wf

    # ------------------------------------------------------------------
    # Matérialisation d'un état invalide
    # ------------------------------------------------------------------

    def _materialise(self, node, wind_field):
        """
        Si node.state_valid est False, recalcule l'état depuis le parent.
        Appelé juste avant toute utilisation de node.state.
        """
        if node.state_valid:
            return
        # Le parent est toujours valide (on s'assure de ça en descendant l'arbre)
        parent_state = node.parent.state
        new_state = _apply_sim_step(
            parent_state, node.action, self.world_map, wind_field, self.mean_rot_rad
        )
        node.state       = new_state
        node.state_valid = True
        # Réinitialiser untried_actions si nécessaire (état inconnu à la création)
        dist_sq = (new_state.px - 64)**2 + (new_state.py - 127)**2
        if not node.untried_actions and not node.children:
            node.untried_actions = list(range(9 if dist_sq < 25 else 8))

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, state, wind_field):
        s = state
        for _ in range(self.rollout_depth):
            if s.done:
                break
            a = _rollout_action(s, self.world_map, wind_field)
            s = _apply_sim_step(s, a, self.world_map, wind_field, self.mean_rot_rad)
        return _terminal_value(s, self.gamma)

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------

    def _run_mcts(self, wind_field):
        c    = self.c_puct
        root = self._root

        for _ in range(self.n_simulations):

            # ── 1. Sélection ──────────────────────────────────────────
            node  = root
            depth = 0
            while depth < self.max_depth:
                # Matérialiser si nécessaire avant de tester done / expanded
                self._materialise(node, wind_field)
                if node.state.done or not node.is_fully_expanded():
                    break
                lp   = math.log(node.n_visits) if node.n_visits > 1 else 0.0
                node = node.best_child(c, lp)
                depth += 1

            self._materialise(node, wind_field)

            # ── 2. Expansion ──────────────────────────────────────────
            if not node.state.done and not node.is_fully_expanded():
                idx    = np.random.randint(len(node.untried_actions))
                action = node.untried_actions.pop(idx)

                # L'état enfant est calculé maintenant, depuis un parent valide,
                # avec le wind_field observé à ce step.
                child_state = _apply_sim_step(
                    node.state, action, self.world_map, wind_field, self.mean_rot_rad
                )
                child = MCTSNode(child_state, parent=node, action=action, state_valid=True)
                node.children[action] = child
                node = child

            # ── 3. Rollout ────────────────────────────────────────────
            value = self._rollout(node.state, wind_field)

            # ── 4. Backprop ───────────────────────────────────────────
            n = node
            while n is not None:
                n.n_visits    += 1
                n.total_value += value
                n = n.parent

    # ------------------------------------------------------------------
    # Tree reuse : invalide les états hérités, conserve les stats UCB1
    # ------------------------------------------------------------------

    def _reuse_subtree(self, best_action, real_root_state):
        """
        Descend dans le sous-arbre de best_action.
        L'état de la nouvelle racine est REMPLACÉ par l'état réel observé
        (qui servira de base pour recalculer les enfants au prochain step).
        Les enfants de la nouvelle racine sont marqués state_valid=False :
        leurs états seront recalculés à la demande depuis le vrai nouvel état.
        """
        new_root = self._root.children[best_action]
        new_root.parent = None

        # On remplace l'état simulé hérité par l'état réel observé
        new_root.state       = real_root_state
        new_root.state_valid = True

        # Invalider les états de tous les enfants directs (profondeur 1)
        # Les enfants plus profonds seront invalidés en cascade via _materialise
        for child in new_root.children.values():
            child.state_valid = False
            # Invalider récursivement (BFS léger, évite la récursion profonde)
            queue = list(child.children.values())
            while queue:
                n = queue.pop()
                n.state_valid = False
                queue.extend(n.children.values())

        self._root = new_root

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> int:
        px, py, vx, vy, wind_field = self._parse_obs(observation)
        self._step_count += 1

        if (px - 64)**2 + (py - 127)**2 < 2.25:
            return 8

        real_state = SimState(
            px=int(round(px)), py=int(round(py)),
            vx=int(round(vx)), vy=int(round(vy)),
            wind_angle_offset=0.0,
            step=self._step_count,
        )

        if self._root is None:
            self._root = MCTSNode(real_state, state_valid=True)
        else:
            # Recaler la racine sur l'observation réelle (sans toucher aux enfants ici)
            self._root.state       = real_state
            self._root.state_valid = True

        self._run_mcts(wind_field)

        if not self._root.children:
            self._root = None
            return int(_rollout_action(real_state, self.world_map, wind_field))

        best_action = self._root.most_visited_action()
        self._reuse_subtree(best_action, real_state)

        return int(best_action)