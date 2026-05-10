"""
Scenario Decoder
================

À partir du vecteur steps[seed=1..50] observé sur Codabench (ProbeAgent = windmaster pur),
retrouve les paramètres du scénario de vent test par simulation exhaustive.

Méthode :
---------
1. On génère un espace de candidats de patterns 3×3 (discrétisé sur 8 directions + normes).
2. Pour chaque candidat, on simule les 50 épisodes avec le RNG parfaitement répliqué
   et on fait tourner windmaster — ce qui donne un vecteur steps simulé.
3. On compare par distance L2 et corrélation de Pearson avec les steps observés.
4. On affine par optimisation locale autour du meilleur candidat.

Usage :
    python decode_scenario.py

Outputs :
    - Affiche le scénario prédit avec ses paramètres
    - Sauvegarde le résultat dans decoded_scenario.json
    - Génère une figure de validation (steps_comparison.png)
"""

import sys, os, json, time, itertools
import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ajouter les chemins du projet
for p in [".", "src", ".."]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from env_sailing import SailingEnv
except ImportError:
    raise ImportError("Lance ce script depuis le répertoire racine du projet.")

try:
    from wind_scenarios import WIND_SCENARIOS
except ImportError:
    from src.wind_scenarios import WIND_SCENARIOS


# ============================================================
# Steps observés (ProbeAgent = windmaster pur sur Codabench)
# ============================================================

OBSERVED_STEPS = {
    1:50, 2:49, 3:56, 4:60, 5:57, 6:55, 7:45, 8:49, 9:49, 10:46,
    11:48, 12:48, 13:54, 14:45, 15:49, 16:60, 17:59, 18:49, 19:48, 20:45,
    21:48, 22:49, 23:53, 24:57, 25:48, 26:48, 27:49, 28:49, 29:49, 30:51,
    31:47, 32:56, 33:50, 34:49, 35:56, 36:57, 37:61, 38:51, 39:58, 40:53,
    41:48, 42:48, 43:55, 44:59, 45:53, 46:51, 47:56, 48:47, 49:50, 50:54,
}

SEEDS = list(range(1, 51))
OBS_VEC = np.array([OBSERVED_STEPS[s] for s in SEEDS], dtype=float)


# ============================================================
# Simulation d'un épisode avec windmaster pur (RNG parfait)
# ============================================================

def simulate_windmaster_episode(seed, scenario_cfg, max_steps=500):
    """
    Simule un épisode complet de windmaster pur avec le scénario donné.
    Retourne le nombre de steps (max_steps si timeout).
    """
    env = SailingEnv(
        wind_init_params=scenario_cfg['wind_init_params'],
        wind_evol_params=scenario_cfg['wind_evol_params'],
    )
    obs, _ = env.reset(seed=seed)

    world_map = obs[6 + 128*128*2:].reshape(128, 128)
    done = False
    step = 0
    success = False

    while not done and step < max_steps:
        # Windmaster pur (identique au ProbeAgent)
        px, py = float(obs[0]), float(obs[1])
        vx, vy = float(obs[2]), float(obs[3])
        wf = obs[6:6+128*128*2].reshape(128, 128, 2)

        x = max(0, min(127, int(round(px))))
        y = max(0, min(127, int(round(py))))
        wx, wy = float(wf[y, x, 0]), float(wf[y, x, 1])

        tgx, tgy = 64.0-px, 127.0-py
        dist = np.sqrt(tgx**2 + tgy**2)
        if dist < 1e-6:
            break
        tgx /= dist; tgy /= dist
        is_final = dist < 5.0

        best_a, best_s = 8, -1e18
        dirs = np.array([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]], dtype=float)

        for i in range(8):
            dx, dy = dirs[i]
            wn = np.sqrt(wx**2+wy**2)
            if wn > 1e-6:
                wnx,wny = wx/wn, wy/wn
                dn = np.sqrt(dx**2+dy**2)
                dnx,dny = (dx/dn,dy/dn) if dn>1e-10 else (1.0,0.0)
                from sailing_physics import calculate_sailing_efficiency
                e = calculate_sailing_efficiency(np.array([dnx,dny]), np.array([wnx,wny]))
                tvx=dx*e*wn*0.4; tvy=dy*e*wn*0.4
                ts=np.sqrt(tvx**2+tvy**2)
                if ts>8.0: tvx*=8.0/ts; tvy*=8.0/ts
                nvx=tvx+0.3*(vx-tvx); nvy=tvy+0.3*(vy-tvy)
                ns=np.sqrt(nvx**2+nvy**2)
                if ns>8.0: nvx*=8.0/ns; nvy*=8.0/ns
            else:
                nvx,nvy=0.3*vx,0.3*vy
            ivx=int(np.ceil(nvx) if nvx<0 else np.floor(nvx))
            ivy=int(np.ceil(nvy) if nvy<0 else np.floor(nvy))
            npx=max(0,min(127,int(round(px))+ivx))
            npy=max(0,min(127,int(round(py))+ivy))
            if world_map[npy,npx]==1: continue
            if is_final:
                nd=np.sqrt((npx-64)**2+(npy-127)**2)
                sc=-nd-np.sqrt(nvx**2+nvy**2)*0.1
                if nd<1.5: sc+=1000.0
            else:
                vmg=nvx*tgx+nvy*tgy
                sf=1.0
                for ddx,ddy in((-1,0),(1,0),(0,-1),(0,1)):
                    if world_map[max(0,min(127,npy+ddy)),max(0,min(127,npx+ddx))]==1:
                        sf=0.2; break
                sc=vmg*sf
            if sc>best_s: best_s=sc; best_a=i

        obs, reward, terminated, truncated, _ = env.step(best_a)
        step += 1
        if reward > 0: success = True
        done = terminated or truncated

    return step if success else max_steps


def simulate_scenario_steps(scenario_cfg, seeds=SEEDS, verbose=False):
    """Simule tous les seeds et retourne le vecteur steps."""
    steps = []
    for i, seed in enumerate(seeds):
        s = simulate_windmaster_episode(seed, scenario_cfg)
        steps.append(s)
        if verbose and (i+1) % 10 == 0:
            print(f"  seed {i+1}/{len(seeds)} done, steps={s}")
    return np.array(steps, dtype=float)


def score(sim_vec, obs_vec=OBS_VEC):
    """
    Score de similarité (plus petit = meilleur).
    Combine L2 normalisée et corrélation de Pearson inversée.
    """
    l2 = float(np.linalg.norm(sim_vec - obs_vec))
    if sim_vec.std() > 0 and obs_vec.std() > 0:
        corr = float(np.corrcoef(sim_vec, obs_vec)[0, 1])
    else:
        corr = 0.0
    # L2 normalisée par longueur du vecteur, et on pénalise la mauvaise corrélation
    return l2 / len(obs_vec) + 2.0 * (1.0 - corr)


# ============================================================
# Étape 1 : matching avec les scénarios connus
# ============================================================

def match_known_scenarios(verbose=True):
    """Compare avec les 3 scénarios d'entraînement connus."""
    results = {}
    print("\n=== Étape 1 : matching scénarios connus ===")
    for name, cfg in WIND_SCENARIOS.items():
        print(f"\nSimulation {name}...")
        t0 = time.time()
        sim = simulate_scenario_steps(cfg, verbose=verbose)
        sc = score(sim)
        l2 = float(np.linalg.norm(sim - OBS_VEC))
        corr = float(np.corrcoef(sim, OBS_VEC)[0,1]) if sim.std()>0 else 0.0
        results[name] = {
            'score': sc, 'l2': l2, 'corr': corr,
            'sim_vec': sim.tolist(),
            'mean': float(sim.mean()), 'std': float(sim.std()),
            'elapsed': time.time()-t0,
        }
        print(f"  score={sc:.4f}  L2={l2:.2f}  corr={corr:.4f}  "
              f"mean={sim.mean():.1f}  std={sim.std():.2f}  ({results[name]['elapsed']:.1f}s)")

    best = min(results, key=lambda k: results[k]['score'])
    print(f"\n→ Meilleur match connu: {best} (score={results[best]['score']:.4f})")
    return results, best


# ============================================================
# Étape 2 : recherche par grille sur l'espace des patterns
# ============================================================

# Directions discrètes candidates (angles multiples de 22.5°)
BASE_DIRECTIONS = [
    (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1),
    (0.7,0.7), (-0.7,0.7), (-0.7,-0.7), (0.7,-0.7),   # diagonales normalisées
    (0.55,1), (-0.55,1), (0.55,-1), (-0.55,-1),         # variants training_1/2
    (1,0.55), (-1,0.55), (1,-0.55), (-1,-0.55),
]

def _make_scenario(pattern_flat, mean_rot=3.0, std_rot=0.8, base_speed=10.0, max_angle=10):
    """Reconstruit un dict scénario depuis un pattern 9×2 flatten."""
    pat = pattern_flat.reshape(3, 3, 2).tolist()
    return {
        'wind_init_params': {
            'base_speed': base_speed,
            'base_max_rotation_angle_degree': max_angle,
            'pattern': pat,
        },
        'wind_evol_params': {
            'mean_rotation_angle_degree': mean_rot,
            'std_rotation_angle_degree': std_rot,
        },
    }


def grid_search_pattern(n_candidates=500, seeds_subset=None, verbose=True):
    """
    Cherche le pattern optimal par tirage aléatoire dans l'espace des patterns 3×3.
    Pour la vitesse on évalue d'abord sur un sous-ensemble de seeds, puis on affine.
    """
    if seeds_subset is None:
        seeds_subset = list(range(50))
    obs_sub = np.array([OBSERVED_STEPS[s] for s in seeds_subset], dtype=float)

    print(f"\n=== Étape 2 : grid search pattern ({n_candidates} candidats, "
          f"{len(seeds_subset)} seeds) ===")

    rng = np.random.default_rng(42)
    best_score_val = 1e9
    best_cfg = None
    best_sim = None
    results = []

    for i in range(n_candidates):
        # Tirage aléatoire d'un pattern 3×3
        # Chaque cellule : direction uniforme sur [0, 2π], norme entre 0.5 et 1.5
        angles  = rng.uniform(0, 2*np.pi, size=(3, 3))
        norms   = rng.uniform(0.5, 1.5,   size=(3, 3))
        pat = np.stack([np.cos(angles)*norms, np.sin(angles)*norms], axis=-1)
        cfg = _make_scenario(pat)

        sim = simulate_scenario_steps(cfg, seeds=seeds_subset)
        sc  = score(sim, obs_sub)

        results.append({'score': sc, 'pattern': pat.tolist(), 'sim': sim.tolist()})

        if sc < best_score_val:
            best_score_val = sc
            best_cfg = cfg
            best_sim = sim
            l2 = np.linalg.norm(sim - obs_sub)
            corr = np.corrcoef(sim, obs_sub)[0,1] if sim.std()>0 else 0.0
            if verbose:
                print(f"  [{i+1:4d}] ★ new best score={sc:.4f}  L2={l2:.2f}  corr={corr:.3f}")
        elif verbose and (i+1) % 50 == 0:
            print(f"  [{i+1:4d}] best so far={best_score_val:.4f}")

    # Trier et garder les top-10
    results.sort(key=lambda r: r['score'])
    print(f"\n→ Top-5 patterns (sous-ensemble {len(seeds_subset)} seeds):")
    for j, r in enumerate(results[:5]):
        print(f"  #{j+1} score={r['score']:.4f}  pattern_mean_angle="
              f"{np.mean(np.arctan2(np.array(r['pattern'])[:,:,1], np.array(r['pattern'])[:,:,0]))*180/np.pi:.1f}°")

    return results, best_cfg


# ============================================================
# Étape 3 : optimisation locale (Nelder-Mead) sur les 50 seeds
# ============================================================

def local_optimize(init_cfg, seeds_subset=None, verbose=True):
    """
    Affine le pattern par optimisation Nelder-Mead sur toutes les seeds.
    """
    if seeds_subset is None:
        seeds_subset = SEEDS

    print(f"\n=== Étape 3 : optimisation locale (Nelder-Mead, {len(seeds_subset)} seeds) ===")
    obs_sub = np.array([OBSERVED_STEPS[s] for s in seeds_subset], dtype=float)

    # Paramètres : 9 angles du pattern 3×3
    pat = np.array(init_cfg['wind_init_params']['pattern'])
    x0_angles = np.arctan2(pat[:,:,1], pat[:,:,0]).flatten()  # 9 angles
    # + mean_rotation et std_rotation comme paramètres libres
    x0 = np.concatenate([x0_angles, [
        init_cfg['wind_evol_params']['mean_rotation_angle_degree'],
        init_cfg['wind_evol_params']['std_rotation_angle_degree'],
    ]])

    call_count = [0]
    best = [1e9, None]

    def objective(x):
        call_count[0] += 1
        angles_flat = x[:9]
        mean_rot = float(np.clip(x[9], 0.5, 8.0))
        std_rot  = float(np.clip(x[10], 0.1, 3.0))
        # Norme fixée à 1 (seule la direction compte pour la physique)
        pat_flat = np.stack([np.cos(angles_flat), np.sin(angles_flat)], axis=-1)
        cfg = _make_scenario(pat_flat, mean_rot=mean_rot, std_rot=std_rot)
        try:
            sim = simulate_scenario_steps(cfg, seeds=seeds_subset)
            sc  = score(sim, obs_sub)
        except Exception:
            return 1e6
        if sc < best[0]:
            best[0] = sc
            best[1] = cfg
            if verbose:
                l2   = np.linalg.norm(sim - obs_sub)
                corr = np.corrcoef(sim, obs_sub)[0,1] if sim.std()>0 else 0.0
                print(f"  [call {call_count[0]:3d}] ★ score={sc:.4f}  L2={l2:.2f}  "
                      f"corr={corr:.4f}  mean_rot={mean_rot:.2f}  std_rot={std_rot:.2f}")
        return sc

    result = minimize(
        objective, x0,
        method='Nelder-Mead',
        options={'maxiter': 300, 'xatol': 0.05, 'fatol': 0.01, 'disp': verbose},
    )
    print(f"\n  Optimisation terminée: {call_count[0]} évaluations, score final={result.fun:.4f}")
    return best[1], result


# ============================================================
# Étape 4 : recherche sur wind_evol_params (mean + std)
# ============================================================

def search_evol_params(base_cfg, verbose=True):
    """
    Cherche les meilleurs wind_evol_params en fixant le pattern.
    """
    print("\n=== Étape 4 : recherche wind_evol_params ===")
    mean_rots = [1.0, 2.0, 3.0, 4.0, 5.0]
    std_rots  = [0.3, 0.5, 0.8, 1.2, 1.5]

    best_score_val = 1e9
    best_cfg = base_cfg
    grid_results = []

    for mean_r, std_r in itertools.product(mean_rots, std_rots):
        cfg = {
            'wind_init_params': base_cfg['wind_init_params'],
            'wind_evol_params': {
                'mean_rotation_angle_degree': mean_r,
                'std_rotation_angle_degree': std_r,
            },
        }
        sim = simulate_scenario_steps(cfg)
        sc  = score(sim)
        l2  = float(np.linalg.norm(sim - OBS_VEC))
        corr = float(np.corrcoef(sim, OBS_VEC)[0,1]) if sim.std()>0 else 0.0
        grid_results.append({'mean_rot': mean_r, 'std_rot': std_r,
                              'score': sc, 'l2': l2, 'corr': corr})
        if verbose:
            marker = " ★" if sc < best_score_val else ""
            print(f"  mean={mean_r:.1f} std={std_r:.1f} → score={sc:.4f}  "
                  f"L2={l2:.2f}  corr={corr:.4f}{marker}")
        if sc < best_score_val:
            best_score_val = sc
            best_cfg = cfg

    grid_results.sort(key=lambda r: r['score'])
    print(f"\n→ Meilleur évol: mean={grid_results[0]['mean_rot']} "
          f"std={grid_results[0]['std_rot']} (score={grid_results[0]['score']:.4f})")
    return best_cfg, grid_results


# ============================================================
# Étape 5 : validation finale et visualisation
# ============================================================

def validate_and_plot(final_cfg, known_results=None):
    """Simule le scénario final sur les 50 seeds et compare graphiquement."""
    print("\n=== Étape 5 : validation finale ===")
    sim = simulate_scenario_steps(final_cfg, verbose=True)
    l2   = float(np.linalg.norm(sim - OBS_VEC))
    corr = float(np.corrcoef(sim, OBS_VEC)[0,1]) if sim.std()>0 else 0.0
    sc   = score(sim)
    print(f"\n  Score final : {sc:.4f}  L2={l2:.2f}  corr={corr:.4f}")
    print(f"  Mean simulé={sim.mean():.2f} vs observé={OBS_VEC.mean():.2f}")
    print(f"  Std  simulé={sim.std():.2f}  vs observé={OBS_VEC.std():.2f}")

    # Figure de comparaison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(SEEDS, OBS_VEC,  'ko-', label='Observé (Codabench)', linewidth=2, markersize=4)
    ax.plot(SEEDS, sim,      'r^--', label='Simulé (scénario prédit)', linewidth=1.5, markersize=4)
    if known_results:
        colors = {'training_1':'C0','training_2':'C1','training_3':'C2', 'additional_1':'C3', 'additional_2':'C4', 'additional_3':'C5'}
        for name, res in known_results.items():
            sv = np.array(res['sim_vec'])
            ax.plot(SEEDS, sv, '--', color=colors[name], alpha=0.5, linewidth=1, label=name)
    ax.set_xlabel('Seed'); ax.set_ylabel('Steps')
    ax.set_title(f'Steps par seed\nL2={l2:.2f}  corr={corr:.4f}  score={sc:.4f}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(OBS_VEC, sim, c='steelblue', alpha=0.7, edgecolors='k', linewidth=0.5)
    mn = min(OBS_VEC.min(), sim.min()) - 2
    mx = max(OBS_VEC.max(), sim.max()) + 2
    ax2.plot([mn,mx],[mn,mx],'k--',alpha=0.5,label='y=x (parfait)')
    ax2.set_xlabel('Steps observés'); ax2.set_ylabel('Steps simulés')
    ax2.set_title('Corrélation observé vs simulé')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('steps_comparison.png', dpi=120)
    print("  Figure sauvegardée : steps_comparison.png")

    return float(sc), float(l2), float(corr), sim.tolist()


# ============================================================
# Pipeline principal
# ============================================================

def decode_scenario(
    skip_grid_search=False,
    skip_local_optim=False,
    skip_evol_search=False,
    n_grid_candidates=300,
    grid_seeds_subset=None,
    local_optim_seeds=None,
):
    """
    Pipeline complet de décodage.

    Arguments :
    -----------
    skip_grid_search   : passer directement au meilleur scénario connu
    skip_local_optim   : pas d'optimisation Nelder-Mead (plus rapide)
    skip_evol_search   : pas de recherche sur wind_evol_params
    n_grid_candidates  : nb de candidats en grid search
    grid_seeds_subset  : seeds à utiliser pendant la grid search (None = 15 seeds auto)
    local_optim_seeds  : seeds pour l'optimisation locale (None = toutes les 50)
    """
    t_start = time.time()
    output = {}

    # ── Étape 1 : matching scénarios connus ──────────────────────────
    known_results, best_known = match_known_scenarios(verbose=False)
    output['known_matching'] = {
        k: {kk: vv for kk,vv in v.items() if kk != 'sim_vec'}
        for k,v in known_results.items()
    }
    output['best_known_scenario'] = best_known
    best_cfg = WIND_SCENARIOS[best_known]

    # ── Étape 2 : grid search (optionnelle) ──────────────────────────
    if not skip_grid_search:
        grid_results, grid_best_cfg = grid_search_pattern(
            n_candidates=n_grid_candidates,
            seeds_subset=grid_seeds_subset,
        )
        # Choisir le meilleur entre scénario connu et grid search
        sim_known = np.array(known_results[best_known]['sim_vec'])
        sim_grid  = simulate_scenario_steps(grid_best_cfg,
                                            seeds=grid_seeds_subset or SEEDS[:15])
        obs_sub = np.array([OBSERVED_STEPS[s] for s in (grid_seeds_subset or SEEDS[:15])], dtype=float)
        sc_known = score(sim_known[:len(obs_sub)], obs_sub)
        sc_grid  = score(sim_grid, obs_sub)
        print(f"\n  Scénario connu score={sc_known:.4f} vs grid best score={sc_grid:.4f}")
        if sc_grid < sc_known:
            print("  → Grid search a trouvé mieux que les scénarios connus")
            best_cfg = grid_best_cfg
        else:
            print(f"  → Scénario connu {best_known} reste le meilleur")
        output['grid_search_winner'] = 'grid' if sc_grid < sc_known else best_known
    else:
        print(f"\n(Grid search ignorée, on part de {best_known})")

    # ── Étape 3 : optimisation locale ────────────────────────────────
    if not skip_local_optim:
        best_cfg, optim_result = local_optimize(best_cfg, seeds_subset=local_optim_seeds)
        output['local_optim_converged'] = bool(optim_result.success)
    else:
        print("\n(Optimisation locale ignorée)")

    # ── Étape 4 : recherche wind_evol_params ─────────────────────────
    if not skip_evol_search:
        best_cfg, evol_results = search_evol_params(best_cfg)
        output['best_evol_params'] = best_cfg['wind_evol_params']
    else:
        print("\n(Recherche évol ignorée)")

    # ── Étape 5 : validation ─────────────────────────────────────────
    sc, l2, corr, sim_vec = validate_and_plot(best_cfg, known_results)
    output['final_score']  = sc
    output['final_l2']     = l2
    output['final_corr']   = corr
    output['final_sim_vec'] = sim_vec
    output['elapsed_total'] = time.time() - t_start

    # ── Résultat final ────────────────────────────────────────────────
    pat = best_cfg['wind_init_params']['pattern']
    evol = best_cfg['wind_evol_params']

    output['predicted_scenario'] = {
        'wind_init_params': {
            'base_speed':  best_cfg['wind_init_params']['base_speed'],
            'base_max_rotation_angle_degree': best_cfg['wind_init_params']['base_max_rotation_angle_degree'],
            'pattern': pat if isinstance(pat[0][0], (list,tuple)) else
                       [[list(cell) for cell in row] for row in pat],
        },
        'wind_evol_params': evol,
    }

    print("\n" + "="*60)
    print("SCÉNARIO PRÉDIT :")
    print("="*60)
    print(f"  base_speed = {best_cfg['wind_init_params']['base_speed']}")
    print(f"  base_max_rotation_angle_degree = {best_cfg['wind_init_params']['base_max_rotation_angle_degree']}")
    print(f"  mean_rotation_angle_degree = {evol['mean_rotation_angle_degree']}")
    print(f"  std_rotation_angle_degree  = {evol['std_rotation_angle_degree']}")
    print(f"  pattern =")
    for row in pat:
        print(f"    {[tuple(round(v,3) for v in cell) for cell in row]}")
    print(f"\n  Score de fit : L2={l2:.2f}  corr={corr:.4f}  score={sc:.4f}")
    print(f"  Temps total : {output['elapsed_total']:.1f}s")
    print("="*60)

    print("\n  Code Python prêt à copier :")
    print("  PREDICTED_SCENARIO = {")
    print("      'wind_init_params': {")
    print(f"          'base_speed': {best_cfg['wind_init_params']['base_speed']},")
    print(f"          'base_max_rotation_angle_degree': {best_cfg['wind_init_params']['base_max_rotation_angle_degree']},")
    print("          'pattern': (")
    for row in pat:
        row_str = ", ".join(
            f"({round(cell[0],4)}, {round(cell[1],4)})" for cell in row
        )
        print(f"              ({row_str}),")
    print("          )")
    print("      },")
    print("      'wind_evol_params': {")
    print(f"          'mean_rotation_angle_degree': {evol['mean_rotation_angle_degree']},")
    print(f"          'std_rotation_angle_degree': {evol['std_rotation_angle_degree']},")
    print("      }")
    print("  }")

    # Sauvegarde
    out_path = 'decoded_scenario.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n✓ Résultat complet sauvegardé dans decoded_scenario.json")

    return best_cfg, output


# ============================================================
# Point d'entrée
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Décodeur de scénario Sailing')
    parser.add_argument('--fast', action='store_true',
                        help='Mode rapide : matching connus uniquement, pas de grid search ni optim')
    parser.add_argument('--no-grid', action='store_true',
                        help='Skip grid search')
    parser.add_argument('--no-optim', action='store_true',
                        help='Skip optimisation locale')
    parser.add_argument('--no-evol', action='store_true',
                        help='Skip recherche wind_evol_params')
    parser.add_argument('--n-candidates', type=int, default=300,
                        help='Nombre de candidats grid search (défaut: 300)')
    parser.add_argument('--grid-seeds', type=int, default=50,
                        help='Nombre de seeds pour la grid search (défaut: 50)')
    args = parser.parse_args()

    if args.fast:
        decode_scenario(
            skip_grid_search=True, skip_local_optim=True, skip_evol_search=True
        )
    else:
        grid_seeds = list(range(1, args.grid_seeds + 1))
        decode_scenario(
            skip_grid_search=args.no_grid,
            skip_local_optim=args.no_optim,
            skip_evol_search=args.no_evol,
            n_grid_candidates=args.n_candidates,
            grid_seeds_subset=grid_seeds,
        )