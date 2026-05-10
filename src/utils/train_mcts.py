"""
Script d'évaluation et de comparaison : MCTS vs Windmaster
===========================================================

Ce script :
1. Évalue l'agent MCTS et l'agent Windmaster baseline sur les 3 scénarios d'entrainement.
2. Collecte toutes les métriques demandées (générales + spécifiques MCTS).
3. Affiche les résultats à intervalle régulier pendant l'évaluation.
4. Sauvegarde tout dans un fichier JSON à la fin.

Usage :
    python train_mcts.py [--n_episodes 50] [--n_simulations 300] [--scenarios training_1,training_2,training_3]

Note : "Entraînement" ici = évaluation sur grille de paramètres MCTS (pas de gradient descent).
       On cherche les meilleurs hyperparamètres (n_simulations, max_depth, rollout_depth).
"""

import json
import time
import argparse
import numpy as np

# ── Imports du projet ────────────────────────────────────────────────────────
from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
from src.agents.mcts_agent import MyAgent as MCTSSailingAgent
from src.agents.windmaster import MyAgent as WindmasterAgent


# ────────────────────────────────────────────────────────────────────────────
# Évaluation d'un épisode
# ────────────────────────────────────────────────────────────────────────────

def run_episode(agent, env, seed=None, verbose=False):
    """
    Lance un épisode complet et retourne un dict de métriques.
    """
    obs, info = env.reset(seed=seed)
    agent.reset()
    if hasattr(agent, 'seed'):
        agent.seed(seed)

    gamma = env.reward_discount_factor  # 0.995
    done = False
    step = 0
    discounted_reward = 0.0
    discount = 1.0
    reached = False
    crashed = False

    t_start = time.time()

    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        discounted_reward += discount * reward
        discount *= gamma
        step += 1

        if reward > 0:
            reached = True
        if info.get("is_stuck", False):
            crashed = True

        done = terminated or truncated

        if verbose and step % 50 == 0:
            print(f"  step={step:3d} | pos={info['position']} | "
                  f"dist={info['distance_to_goal']:.1f} | "
                  f"vel={info['velocity']}")

    elapsed = time.time() - t_start

    return {
        "reached": reached,
        "crashed": crashed,
        "steps": step if reached else None,
        "discounted_reward": discounted_reward,
        "elapsed_s": elapsed,
    }


# ────────────────────────────────────────────────────────────────────────────
# Évaluation multi-épisodes
# ────────────────────────────────────────────────────────────────────────────

def evaluate_agent(agent, env, n_episodes=50, log_interval=10, label="Agent"):
    """
    Évalue un agent sur n_episodes épisodes et retourne les métriques agrégées.
    Affiche un résumé toutes les log_interval épisodes.
    """
    results = []
    t0 = time.time()

    for ep in range(n_episodes):
        seed = ep  # seeds reproductibles
        res = run_episode(agent, env, seed=seed)
        results.append(res)

        # Log intermédiaire
        if (ep + 1) % log_interval == 0 or ep == 0:
            recent = results[max(0, ep + 1 - log_interval):]
            success_rate = np.mean([r["reached"] for r in recent])
            avg_score = np.mean([r["discounted_reward"] for r in recent])
            crash_rate = np.mean([r["crashed"] for r in recent])
            steps_ok = [r["steps"] for r in recent if r["steps"] is not None]
            avg_steps = np.mean(steps_ok) if steps_ok else float("nan")
            elapsed = time.time() - t0

            print(
                f"[{label}] ep {ep+1:3d}/{n_episodes} | "
                f"success={success_rate*100:.1f}% | "
                f"score={avg_score:.2f} | "
                f"crash={crash_rate*100:.1f}% | "
                f"steps={avg_steps:.1f} | "
                f"total_time={elapsed:.1f}s"
            )

    # Métriques générales
    success_rate = np.mean([r["reached"] for r in results])
    avg_score = np.mean([r["discounted_reward"] for r in results])
    crash_rate = np.mean([r["crashed"] for r in results])
    steps_ok = [r["steps"] for r in results if r["steps"] is not None]
    avg_steps = float(np.mean(steps_ok)) if steps_ok else float("nan")
    std_steps = float(np.std(steps_ok)) if steps_ok else float("nan")
    avg_time_per_ep = float(np.mean([r["elapsed_s"] for r in results]))

    metrics = {
        # ── Métriques générales ──────────────────────────────────────
        "success_rate": float(success_rate),
        "avg_discounted_score": float(avg_score),
        "avg_steps_on_success": avg_steps,
        "std_steps_on_success": std_steps,
        # ── Métriques spécifiques ────────────────────────────────────
        "crash_rate": float(crash_rate),
        "timeout_rate": float(np.mean([
            not r["reached"] and not r["crashed"] for r in results
        ])),
        "avg_time_per_episode_s": avg_time_per_ep,
        "n_episodes": n_episodes,
        # ── Distributions ────────────────────────────────────────────
        "score_distribution": [r["discounted_reward"] for r in results],
        "steps_distribution": [r["steps"] if r["steps"] else -1 for r in results],
    }
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Recherche d'hyperparamètres (grid search léger)
# ────────────────────────────────────────────────────────────────────────────

def hyperparameter_search(scenario_name="training_1", n_episodes_per_config=20,
                          log_interval=5):
    """
    Lance une grid search sur les hyperparamètres MCTS clés.
    Retourne le meilleur config.
    """
    configs = [
        {"n_simulations": 100, "max_depth": 10, "rollout_depth": 20},
        {"n_simulations": 200, "max_depth": 15, "rollout_depth": 25},
        {"n_simulations": 300, "max_depth": 15, "rollout_depth": 30},
        {"n_simulations": 500, "max_depth": 20, "rollout_depth": 35},
    ]

    print(f"\n{'='*60}")
    print(f"Grid Search MCTS — scénario: {scenario_name}")
    print(f"{'='*60}")

    scenario_params = get_wind_scenario(scenario_name)
    best_config = None
    best_score = -np.inf
    all_results = []

    for cfg in configs:
        print(f"\n→ Config: {cfg}")
        env = SailingEnv(**scenario_params)
        agent = MCTSSailingAgent(**cfg, mean_rotation=3.0, gamma=0.995)
        label = f"MCTS n={cfg['n_simulations']} d={cfg['max_depth']} r={cfg['rollout_depth']}"
        metrics = evaluate_agent(agent, env, n_episodes=n_episodes_per_config,
                                 log_interval=log_interval, label=label)
        cfg_result = {"config": cfg, "metrics": {
            k: v for k, v in metrics.items()
            if k not in ("score_distribution", "steps_distribution")
        }}
        all_results.append(cfg_result)

        score = metrics["success_rate"] * 100 + metrics["avg_discounted_score"]
        print(f"  → Score composite: {score:.2f}")

        if score > best_score:
            best_score = score
            best_config = cfg

    print(f"\n✓ Meilleure config: {best_config} (score composite: {best_score:.2f})")
    return best_config, all_results


# ────────────────────────────────────────────────────────────────────────────
# Évaluation finale complète
# ────────────────────────────────────────────────────────────────────────────

def full_evaluation(mcts_params, n_episodes=50, scenarios=None, log_interval=10):
    """
    Évalue MCTS et Windmaster sur tous les scénarios demandés.
    Sauvegarde les résultats dans results_mcts.json.
    """
    if scenarios is None:
        scenarios = ["training_1", "training_2", "training_3"]

    all_results = {}

    for scenario_name in scenarios:
        print(f"\n{'='*60}")
        print(f"Évaluation finale — scénario: {scenario_name}")
        print(f"{'='*60}")

        scenario_params = get_wind_scenario(scenario_name)

        # ── MCTS ──────────────────────────────────────────────────────
        print(f"\n[MCTS] Paramètres: {mcts_params}")
        env_mcts = SailingEnv(**scenario_params)
        agent_mcts = MCTSSailingAgent(**mcts_params)
        metrics_mcts = evaluate_agent(
            agent_mcts, env_mcts, n_episodes=n_episodes,
            log_interval=log_interval, label=f"MCTS/{scenario_name}"
        )

        # ── Windmaster baseline ────────────────────────────────────────
        metrics_wm = None
        if WindmasterAgent is not None:
            print(f"\n[Windmaster] Baseline")
            env_wm = SailingEnv(**scenario_params)
            agent_wm = WindmasterAgent()
            metrics_wm = evaluate_agent(
                agent_wm, env_wm, n_episodes=n_episodes,
                log_interval=log_interval, label=f"Windmaster/{scenario_name}"
            )

            # Comparaison
            print(f"\n  ── Comparaison {scenario_name} ──")
            print(f"  {'Métrique':<30} {'MCTS':>10} {'Windmaster':>12} {'Delta':>10}")
            print(f"  {'-'*64}")
            for key in ["success_rate", "avg_discounted_score",
                        "avg_steps_on_success", "crash_rate"]:
                v_mcts = metrics_mcts[key]
                v_wm = metrics_wm[key]
                if isinstance(v_mcts, float) and not np.isnan(v_mcts):
                    delta = v_mcts - v_wm
                    print(f"  {key:<30} {v_mcts:>10.3f} {v_wm:>12.3f} {delta:>+10.3f}")
                else:
                    print(f"  {key:<30} {'N/A':>10} {'N/A':>12}")

        all_results[scenario_name] = {
            "mcts": {k: v for k, v in metrics_mcts.items()
                     if k not in ("score_distribution", "steps_distribution")},
            "windmaster": {k: v for k, v in metrics_wm.items()
                          if k not in ("score_distribution", "steps_distribution")
                          } if metrics_wm else None,
            "mcts_params": mcts_params,
        }

    # ── Sauvegarde ────────────────────────────────────────────────────
    output_path = "results_mcts.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Résultats sauvegardés dans: {output_path}")

    return all_results


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Évaluation MCTS Sailing Agent")
    parser.add_argument("--n_episodes", type=int, default=50,
                        help="Nombre d'épisodes d'évaluation finale (défaut: 50)")
    parser.add_argument("--n_simulations", type=int, default=None,
                        help="Forcer n_simulations MCTS (bypass grid search)")
    parser.add_argument("--max_depth", type=int, default=15,
                        help="Profondeur max MCTS (défaut: 15)")
    parser.add_argument("--rollout_depth", type=int, default=30,
                        help="Profondeur rollout windmaster (défaut: 30)")
    parser.add_argument("--c_puct", type=float, default=1.414,
                        help="Facteur d'exploration (défaut: 1.414)")
    parser.add_argument("--scenarios", type=str,
                        default="training_1,training_2,training_3",
                        help="Scénarios à évaluer (défaut: tous les 3)")
    parser.add_argument("--grid_search", action="store_true",
                        help="Activer la grid search d'hyperparamètres")
    parser.add_argument("--gs_scenario", type=str, default="training_1",
                        help="Scénario pour la grid search (défaut: training_1)")
    parser.add_argument("--gs_episodes", type=int, default=20,
                        help="Episodes par config lors de la grid search (défaut: 20)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Intervalle de log (défaut: 10)")
    parser.add_argument("--mean_rotation", type=float, default=3.0,
                        help="Rotation moyenne du vent par step en degrés (défaut: 3.0)")
    args = parser.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",")]

    # ── Étape 1 : Grid search optionnelle ─────────────────────────────
    if args.grid_search and args.n_simulations is None:
        best_config, gs_results = hyperparameter_search(
            scenario_name=args.gs_scenario,
            n_episodes_per_config=args.gs_episodes,
            log_interval=max(1, args.gs_episodes // 4),
        )
        # Sauvegarder grid search
        with open("results_grid_search.json", "w") as f:
            json.dump(gs_results, f, indent=2, default=str)
        print(f"\n✓ Grid search sauvegardée dans: results_grid_search.json")
        mcts_params = {**best_config, "mean_rotation": args.mean_rotation, "gamma": 0.995}
    else:
        # Config manuelle ou défaut
        n_sim = args.n_simulations if args.n_simulations else 300
        mcts_params = {
            "n_simulations": n_sim,
            "max_depth": args.max_depth,
            "rollout_depth": args.rollout_depth,
            "mean_rotation": args.mean_rotation,
            "gamma": 0.995,
            "c_puct": args.c_puct,
        }

    print(f"\n→ Paramètres MCTS retenus: {mcts_params}")

    # ── Étape 2 : Évaluation finale ────────────────────────────────────
    full_evaluation(
        mcts_params=mcts_params,
        n_episodes=args.n_episodes,
        scenarios=scenarios,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()