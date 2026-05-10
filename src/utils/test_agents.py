"""
Test des agents Idée 1 et Idée 2 vs Windmaster baseline.

Usage :
    python test_agents.py [--scenario training_1] [--n_seeds 50]

Ce script simule exactement ce que Codabench fait :
- Évalue les seeds 1 à n_seeds séquentiellement sur le même scénario.
- Pour WindHackerAgent, l'état est persistant entre les seeds (comme en eval).
- Affiche les métriques détaillées à chaque seed + résumé final.
"""

import sys, os, argparse, json, time
import numpy as np

try:
    from env_sailing import SailingEnv
except ImportError:
    from src.env_sailing import SailingEnv

try:
    from wind_scenarios import get_wind_scenario
except ImportError:
    from src.wind_scenarios import get_wind_scenario

from src.agents.scenario_aware import ScenarioAwareAgent
from src.agents.wind_hacker import WindHackerAgent, _GlobalState

try:
    from windmaster_agent import MyAgent as WindmasterAgent
    HAS_WINDMASTER = True
except ImportError:
    HAS_WINDMASTER = False
    print("[WARN] windmaster_agent.py non trouvé")


def run_seeds(agent, env, seeds, label="Agent", log_every=5):
    """Évalue l'agent sur une liste de seeds séquentiellement."""
    results = []
    for i, seed in enumerate(seeds):
        obs, _ = env.reset(seed=seed)
        agent.reset()

        done = False
        step = 0
        gamma = 0.995
        discount = 1.0
        reward_sum = 0.0
        reached = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += discount * reward
            discount *= gamma
            step += 1
            if reward > 0: reached = True
            done = terminated or truncated

        # Infos agent hackeur
        extra = ""
        # if hasattr(agent, 'scenario') and agent.scenario:
        #     extra = f" | scenario={agent.scenario} ({agent.confidence:.2f}) corridor={agent.corridor}"

        if (i+1) % log_every == 0 or i == 0:
            print(f"[{label}] seed={seed:2d} ({i+1:2d}/{len(seeds)}) "
                  f"steps={step:3d} success={reached} "
                  f"score={reward_sum:.2f}{extra}")

        results.append({
            "seed": seed,
            "steps": step if reached else None,
            "reached": reached,
            "score": reward_sum,
        })

    success_rate = np.mean([r["reached"] for r in results])
    ok_steps = [r["steps"] for r in results if r["steps"]]
    avg_steps = float(np.mean(ok_steps)) if ok_steps else float("nan")
    avg_score = float(np.mean([r["score"] for r in results]))

    print(f"\n[{label}] ── Résumé ──────────────────────────────────")
    print(f"  Success rate : {success_rate*100:.1f}%")
    print(f"  Avg score    : {avg_score:.4f}")
    print(f"  Avg steps    : {avg_steps:.1f}")
    print()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="training_1",
                        choices=["training_1","training_2","training_3"])
    parser.add_argument("--n_seeds", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--skip_windmaster", action="store_true")
    args = parser.parse_args()

    seeds = list(range(1, args.n_seeds + 1))
    params = get_wind_scenario(args.scenario)

    all_results = {}

    # ── Windmaster baseline ────────────────────────────────────────────
    if HAS_WINDMASTER and not args.skip_windmaster:
        print(f"\n{'='*55}")
        print(f"Windmaster Baseline — {args.scenario}")
        print(f"{'='*55}")
        env = SailingEnv(**params)
        agent = WindmasterAgent()
        all_results["windmaster"] = run_seeds(
            agent, env, seeds, label="Windmaster", log_every=args.log_every
        )

    # ── Idée 1 : ScenarioAwareAgent ────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Idée 1 : ScenarioAwareAgent — {args.scenario}")
    print(f"{'='*55}")
    env = SailingEnv(**params)
    agent = ScenarioAwareAgent(id_threshold=0.85)
    all_results["scenario_aware"] = run_seeds(
        agent, env, seeds, label="ScenarioAware", log_every=args.log_every
    )

    # ── Idée 2 : WindHackerAgent ───────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Idée 2 : WindHackerAgent — {args.scenario}")
    print(f"{'='*55}")
    print("  (Simule l'évaluation Codabench : état persistant entre seeds)")
    _GlobalState.reset_all()  # remet à zéro pour un test propre
    env = SailingEnv(**params)
    agent = WindHackerAgent(id_threshold=0.80, exploration_phase=5, reset_state=True)
    all_results["wind_hacker"] = run_seeds(
        agent, env, seeds, label="WindHacker", log_every=args.log_every
    )

    print(f"\n  Scénario reconstruit: {agent.scenario} "
          f"(confiance={agent.confidence:.3f}), corridor={agent.corridor}")
    if agent.mean_wind_field is not None:
        mwf = agent.mean_wind_field
        print(f"  Vent moyen zone départ (x=40-88, y=0-40): "
              f"wx={np.mean(mwf[0:40,40:88,0]):.2f} wy={np.mean(mwf[0:40,40:88,1]):.2f}")

    # ── Comparaison finale ─────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Comparaison finale — {args.scenario}")
    print(f"{'='*55}")
    print(f"  {'Agent':<20} {'Success%':>8} {'AvgScore':>10} {'AvgSteps':>10}")
    print(f"  {'-'*50}")
    for name, res in all_results.items():
        sr   = np.mean([r["reached"] for r in res]) * 100
        sc   = np.mean([r["score"] for r in res])
        ok   = [r["steps"] for r in res if r["steps"]]
        st   = np.mean(ok) if ok else float("nan")
        print(f"  {name:<20} {sr:>7.1f}% {sc:>10.4f} {st:>10.1f}")

    # Sauvegarde
    with open("results_agents.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Résultats sauvegardés dans results_agents.json")


if __name__ == "__main__":
    main()