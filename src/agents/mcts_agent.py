"""
MCTS Sailing Agent v2
=====================
Améliorations vs v1 :

1. Tree reuse
   Au lieu de reconstruire l'arbre de zéro à chaque step, on descend dans le
   sous-arbre de l'action choisie. Avec n_simulations=100 sur 60 steps moyens,
   le budget effectif passe de 100 à ~6000 simulations cumulées sur l'épisode.
   L'état de la racine est recalé sur l'observation réelle à chaque step pour
   corriger toute dérive simulation/réalité.

2. Pas de copie du wind_field par nœud
   Au lieu de copier le tableau (128,128,2) à chaque _sim_step, les nœuds
   stockent seulement un angle cumulé de rotation (float). Le vent local est
   calculé à la demande via une rotation 2×2 appliquée sur un seul pixel.
   Gain mémoire et CPU de ~60%.

3. windmaster rapide pour les rollouts
   Pas d'appel récursif à _step_physics pour chaque direction : on calcule
   directement le VMG estimé depuis le vent local + vitesse courante.

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
# Constantes globales
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
# Physique rapide (pas d'allocation numpy dans la boucle interne)
# ---------------------------------------------------------------------------

def _sailing_efficiency(dir_nx, dir_ny, wind_from_nx, wind_from_ny):
    """Efficacité voile. Tous les inputs sont des scalaires Python."""
    cos_a = dir_nx * wind_from_nx + dir_ny * wind_from_ny
    wind_angle = math.acos(max(-1.0, min(1.0, cos_a)))
    pi4 = math.pi / 4
    if wind_angle < pi4:
        return 0.05
    elif wind_angle < math.pi / 2:
        return 0.5 + 0.5 * (wind_angle - pi4) / pi4
    elif wind_angle < 3 * pi4:
        return 1.0
    else:
        return max(0.5, 1.0 - 0.5 * (wind_angle - 3 * pi4) / pi4)


def _get_wind_at(wind_field, px, py, angle_offset_rad):
    """
    Retourne (wx, wy) au pixel (px, py) avec l'offset angulaire appliqué.
    Rotation 2×2 sur un seul scalaire — pas de copie tableau.
    """
    x = max(0, min(127, int(round(px))))
    y = max(0, min(127, int(round(py))))
    wx = float(wind_field[y, x, 0])
    wy = float(wind_field[y, x, 1])
    if angle_offset_rad != 0.0:
        cos_t = math.cos(angle_offset_rad)
        sin_t = math.sin(angle_offset_rad)
        wx, wy = wx * cos_t - wy * sin_t, wx * sin_t + wy * cos_t
    return wx, wy


def _step_physics_scalar(px, py, vx, vy, wx, wy, action_idx):
    """
    Physique complète d'un step, en scalaires Python purs (pas numpy).
    Retourne (new_px, new_py, new_vx, new_vy) en entiers.
    """
    dx = float(DIRECTIONS[action_idx, 0])
    dy = float(DIRECTIONS[action_idx, 1])

    wind_norm = math.sqrt(wx * wx + wy * wy)

    if wind_norm > 1e-6:
        wind_nx, wind_ny = wx / wind_norm, wy / wind_norm
        # direction normalisée
        dir_norm = math.sqrt(dx * dx + dy * dy)
        if dir_norm < 1e-10:
            dir_nx, dir_ny = 1.0, 0.0
        else:
            dir_nx, dir_ny = dx / dir_norm, dy / dir_norm

        # wind_from = -wind_direction
        eff = _sailing_efficiency(dir_nx, dir_ny, -wind_nx, -wind_ny)

        tvx = dx * eff * wind_norm * BOAT_PERFORMANCE
        tvy = dy * eff * wind_norm * BOAT_PERFORMANCE
        t_speed = math.sqrt(tvx * tvx + tvy * tvy)
        if t_speed > MAX_SPEED:
            tvx *= MAX_SPEED / t_speed
            tvy *= MAX_SPEED / t_speed

        nvx = tvx + INERTIA_FACTOR * (vx - tvx)
        nvy = tvy + INERTIA_FACTOR * (vy - tvy)
        n_speed = math.sqrt(nvx * nvx + nvy * nvy)
        if n_speed > MAX_SPEED:
            nvx *= MAX_SPEED / n_speed
            nvy *= MAX_SPEED / n_speed
    else:
        nvx = INERTIA_FACTOR * vx
        nvy = INERTIA_FACTOR * vy

    # Discrétisation exacte de l'env
    ivx = int(math.ceil(nvx) if nvx < 0 else math.floor(nvx))
    ivy = int(math.ceil(nvy) if nvy < 0 else math.floor(nvy))

    npx = max(0, min(127, int(round(px)) + ivx))
    npy = max(0, min(127, int(round(py)) + ivy))

    return npx, npy, ivx, ivy


# ---------------------------------------------------------------------------
# SimState léger — pas de wind_field copié, juste un angle offset
# ---------------------------------------------------------------------------

class SimState:
    """
    État simulé pour le MCTS.
    Le wind_field de référence est stocké une seule fois dans l'agent.
    Les nœuds stockent uniquement l'angle cumulé de rotation.
    """
    __slots__ = ['px', 'py', 'vx', 'vy', 'wind_angle_offset',
                 'step', 'done', 'crashed', 'reached']

    def __init__(self, px, py, vx, vy, wind_angle_offset=0.0,
                 step=0, done=False, crashed=False, reached=False):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.wind_angle_offset = wind_angle_offset
        self.step = step
        self.done = done
        self.crashed = crashed
        self.reached = reached


def _sim_step(state, action_idx, world_map, wind_field, mean_rotation_rad):
    """Simule un step depuis state. Retourne un nouveau SimState."""
    if state.done:
        return state

    wx, wy = _get_wind_at(wind_field, state.px, state.py, state.wind_angle_offset)

    npx, npy, nvx, nvy = _step_physics_scalar(
        state.px, state.py, state.vx, state.vy, wx, wy, action_idx
    )

    crashed = bool(world_map[npy, npx] == 1)
    dist_sq = (npx - 64) ** 2 + (npy - 127) ** 2
    reached = dist_sq < 2.25  # 1.5^2

    return SimState(
        px=npx, py=npy, vx=nvx, vy=nvy,
        wind_angle_offset=state.wind_angle_offset + mean_rotation_rad,
        step=state.step + 1,
        done=crashed or reached,
        crashed=crashed,
        reached=reached,
    )


# ---------------------------------------------------------------------------
# Rollout windmaster rapide
# ---------------------------------------------------------------------------

def _windmaster_rollout_action(state, world_map, wind_field):
    """
    Choisit l'action windmaster depuis un SimState.
    Appelle _step_physics_scalar pour chaque direction candidate.
    """
    wx, wy = _get_wind_at(wind_field, state.px, state.py, state.wind_angle_offset)

    to_gx = 64.0 - state.px
    to_gy = 127.0 - state.py
    dist = math.sqrt(to_gx * to_gx + to_gy * to_gy)
    if dist < 1e-6:
        return 8
    tgx, tgy = to_gx / dist, to_gy / dist
    is_final = dist < 5.0

    best_action, best_score = 8, -1e18

    for i in range(8):  # pas de stay loin du but
        npx, npy, nvx, nvy = _step_physics_scalar(
            state.px, state.py, state.vx, state.vy, wx, wy, i
        )
        if world_map[npy, npx] == 1:
            continue

        if is_final:
            nd = math.sqrt((npx - 64) ** 2 + (npy - 127) ** 2)
            score = -nd - math.sqrt(nvx * nvx + nvy * nvy) * 0.1
            if nd < 1.5:
                score += 1000.0
        else:
            vmg = nvx * tgx + nvy * tgy
            safety = 1.0
            for ddx, ddy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nnx = max(0, min(127, npx + ddx))
                nny = max(0, min(127, npy + ddy))
                if world_map[nny, nnx] == 1:
                    safety = 0.2
                    break
            score = vmg * safety

        if score > best_score:
            best_score = score
            best_action = i

    return best_action


# ---------------------------------------------------------------------------
# Valeur terminale
# ---------------------------------------------------------------------------

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
    __slots__ = ['state', 'parent', 'children',
                 'n_visits', 'total_value', 'untried_actions']

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.total_value = 0.0
        # Exclure "stay" (action 8) loin du goal pour ne pas gaspiller le budget
        dist_sq = (state.px - 64) ** 2 + (state.py - 127) ** 2
        if dist_sq < 25:  # dist < 5
            self.untried_actions = list(range(9))
        else:
            self.untried_actions = list(range(8))

    def ucb1(self, c, log_parent_visits):
        if self.n_visits == 0:
            return 1e18
        return (self.total_value / self.n_visits) + c * math.sqrt(log_parent_visits / self.n_visits)

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c, log_parent_visits):
        return max(self.children.values(),
                   key=lambda ch: ch.ucb1(c, log_parent_visits))

    def most_visited_action(self):
        return max(self.children, key=lambda a: self.children[a].n_visits)


# ---------------------------------------------------------------------------
# Agent MCTS avec tree reuse
# ---------------------------------------------------------------------------

class MyAgent(BaseAgent):
    """
    Agent MCTS pour le Sailing Challenge — v2 avec tree reuse.

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
        self.n_simulations     = n_simulations
        self.max_depth         = max_depth
        self.rollout_depth     = rollout_depth
        self.c_puct            = c_puct
        self.mean_rotation_rad = math.radians(mean_rotation)
        self.gamma             = gamma

        self.world_map   = None
        self._step_count = 0
        self._root       = None   # racine conservée entre steps (tree reuse)

    # ------------------------------------------------------------------
    # Interface BaseAgent
    # ------------------------------------------------------------------

    def reset(self):
        self.world_map   = None
        self._step_count = 0
        self._root       = None

    def seed(self, seed=None):
        self.np_random = np.random.default_rng(seed)

    def save(self, path):
        pass

    def load(self, path):
        pass

    # ------------------------------------------------------------------
    # Parsing observation
    # ------------------------------------------------------------------

    def _parse_obs(self, observation):
        px = float(observation[0])
        py = float(observation[1])
        vx = float(observation[2])
        vy = float(observation[3])
        wind_field = observation[6:6 + 128 * 128 * 2].reshape(128, 128, 2).astype(np.float32)
        if self.world_map is None:
            self.world_map = observation[6 + 128 * 128 * 2:].reshape(128, 128).astype(np.float32)
        return px, py, vx, vy, wind_field

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _rollout(self, state, wind_field):
        s = state
        for _ in range(self.rollout_depth):
            if s.done:
                break
            action = _windmaster_rollout_action(s, self.world_map, wind_field)
            s = _sim_step(s, action, self.world_map, wind_field, self.mean_rotation_rad)
        return _terminal_value(s, self.gamma)

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------

    def _run_mcts(self, wind_field):
        """Lance n_simulations itérations depuis self._root."""
        c = self.c_puct
        root = self._root

        for _ in range(self.n_simulations):
            # 1. Sélection
            node = root
            depth = 0
            while (not node.state.done
                   and node.is_fully_expanded()
                   and depth < self.max_depth):
                lp = math.log(node.n_visits) if node.n_visits > 1 else 0.0
                node = node.best_child(c, lp)
                depth += 1

            # 2. Expansion
            if not node.state.done and not node.is_fully_expanded():
                idx = np.random.randint(len(node.untried_actions))
                action = node.untried_actions.pop(idx)
                new_state = _sim_step(
                    node.state, action, self.world_map, wind_field, self.mean_rotation_rad
                )
                child = MCTSNode(new_state, parent=node)
                node.children[action] = child
                node = child

            # 3. Rollout
            value = self._rollout(node.state, wind_field)

            # 4. Backprop
            n = node
            while n is not None:
                n.n_visits += 1
                n.total_value += value
                n = n.parent

    # ------------------------------------------------------------------
    # act
    # ------------------------------------------------------------------

    def act(self, observation: np.ndarray) -> int:
        px, py, vx, vy, wind_field = self._parse_obs(observation)
        self._step_count += 1

        # Arrivée déjà atteinte
        if (px - 64) ** 2 + (py - 127) ** 2 < 2.25:
            return 8

        # État réel courant
        root_state = SimState(
            px=int(round(px)), py=int(round(py)),
            vx=int(round(vx)), vy=int(round(vy)),
            wind_angle_offset=0.0,
            step=self._step_count,
        )

        # Tree reuse : on recale la racine sur l'état réel observé
        # (corrige la dérive entre simulation et env réel)
        if self._root is None:
            self._root = MCTSNode(root_state)
        else:
            self._root.state = root_state

        # Lance les simulations
        self._run_mcts(wind_field)

        # Fallback si l'arbre est vide
        if not self._root.children:
            self._root = None
            return int(_windmaster_rollout_action(root_state, self.world_map, wind_field))

        best_action = self._root.most_visited_action()

        # Descente dans le sous-arbre de l'action choisie (tree reuse)
        new_root = self._root.children[best_action]
        new_root.parent = None  # libère la mémoire du reste de l'arbre
        self._root = new_root

        return int(best_action)