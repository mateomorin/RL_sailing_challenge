"""
Fine-tuning PPO-BC — Scénarios de vent aléatoires
===================================================

Ce script charge les poids existants d'un agent PPO-BC entraîné sur les 3
scénarios de base, puis continue l'entraînement sur une infinité de scénarios
générés aléatoirement.

Contraintes des scénarios générés :
  - Pattern 3×3, chaque tuple ∈ [-1, 1]² (cohérent avec ADDITIONAL_1)
  - Les tuples ne sont PAS normalisés (identique aux scénarios de base)
  - wind_evol_params identiques : mean=3°, std=0.8°
  - base_speed=10.0, base_max_rotation_angle_degree=10 (inchangé)
  - À chaque reset d'un env, un nouveau scénario est tiré → diversité max

Stratégie de fine-tuning :
  - LR réduit (par défaut 3e-5, soit ~10x moins que l'entraînement initial)
  - Entropy bonus légèrement réduit pour préserver la politique acquise
  - Pas de BC (les poids sont déjà bons, on ne veut pas overrider)
  - Toujours 20% de scénarios originaux pour ne pas oublier (catastrophic forgetting)

Utilisation :
  python finetune_random_scenarios.py --weights_in  ppo_bc.npz
                                      --weights_out ppo_bc_finetuned.npz
                                      --steps       2_000_000
                                      --n_envs      20

Arguments optionnels :
  --lr            float   Learning rate (défaut 3e-5)
  --n_envs        int     Nombre d'envs parallèles (défaut 20)
  --steps         int     Total steps de fine-tuning (défaut 2_000_000)
  --rollout       int     Steps par rollout (défaut 512)
  --log_interval  int     Log toutes les N épisodes (défaut 100)
  --orig_ratio    float   Fraction d'envs sur scénarios originaux (défaut 0.2)
  --seed          int     Graine globale pour la reproductibilité (défaut 42)
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  IMPORT DU CODE DE BASE  (agent_ppo_bc.py doit être dans le même dossier)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from src.agents.agent_ppo_bc import (
        extract_features,
        shaped_reward,
        _make_network,
        N_FEATURES,
    )
except ImportError as e:
    raise ImportError(
        "Impossible d'importer agent_ppo_bc.py. "
        "Assurez-vous qu'il est dans le même répertoire.\n"
        f"Erreur : {e}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATEUR DE SCÉNARIOS ALÉATOIRES
# ═══════════════════════════════════════════════════════════════════════════════

# Scénarios de base — gardés pour éviter le catastrophic forgetting
ORIGINAL_SCENARIOS = ('training_1', 'training_2', 'training_3')

# Paramètres fixes communs à tous les scénarios
_FIXED_WIND_INIT = {
    'base_speed': 10.0,
    'base_max_rotation_angle_degree': 10,
}
_FIXED_WIND_EVOL = {
    'mean_rotation_angle_degree': 3,
    'std_rotation_angle_degree': 0.8,
}


def _save_weights(model, path, metrics=None):
    """Sauvegarde les poids du modèle en numpy (.npz) pour inférence sans torch."""
    import s3fs
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
        token=os.environ["AWS_SESSION_TOKEN"]
    )
    weights = {}
    for name, param in model.state_dict().items():
        weights[name] = param.cpu().numpy()
    if metrics:
        weights['_metrics_json'] = np.array([json.dumps(metrics)])
    with fs.open("mamorin/rl_sailing/models/" + path, 'wb') as f:
        np.savez(f, **weights)


def _random_tuple(rng: np.random.Generator) -> tuple:
    """
    Génère un tuple (vx, vy) avec des valeurs dans [-1, 1].

    On utilise une distribution uniforme sur le disque unité
    (direction aléatoire, amplitude entre 0.4 et 1.0) pour éviter
    les vents trop proches de zéro qui donnent des comportements dégénérés,
    tout en couvrant toutes les directions.
    """
    angle = rng.uniform(0, 2 * np.pi)
    # Amplitude entre 0.4 et 1.0 pour des vents "réalistes"
    amp   = rng.uniform(0.4, 1.0)
    vx = float(np.round(amp * np.cos(angle), 4))
    vy = float(np.round(amp * np.sin(angle), 4))
    # Clipping strict à [-1, 1] par sécurité
    vx = float(np.clip(vx, -1.0, 1.0))
    vy = float(np.clip(vy, -1.0, 1.0))
    return (vx, vy)


def generate_random_scenario(rng: np.random.Generator) -> dict:
    """
    Génère un scénario de vent aléatoire avec un pattern 3×3.

    Le pattern est une matrice 3×3 de tuples (vx, vy) ∈ [-1,1]².
    On ajoute une légère cohérence spatiale (les voisins ont des directions
    similaires avec un bruit) pour que le champ de vent soit plus réaliste
    et exploitable par l'agent.

    Returns
    -------
    dict avec 'wind_init_params' et 'wind_evol_params'
    """
    # Générer une direction de base commune (cohérence globale du vent)
    base_angle = rng.uniform(0, 2 * np.pi)
    base_amp   = rng.uniform(0.5, 1.0)

    pattern = []
    for row in range(3):
        row_tuples = []
        for col in range(3):
            # Bruit spatial autour de la direction de base
            noise_angle = rng.normal(0, np.pi / 3)   # ±60° de déviation std
            noise_amp   = rng.uniform(-0.4, 0.4)

            angle = base_angle + noise_angle
            amp   = float(np.clip(base_amp + noise_amp, 0.4, 1.0))

            vx = float(np.clip(np.round(amp * np.cos(angle), 2), -1.0, 1.0))
            vy = float(np.clip(np.round(amp * np.sin(angle), 2), -1.0, 1.0))
            row_tuples.append((vx, vy))
        pattern.append(tuple(row_tuples))

    return {
        'wind_init_params': {
            **_FIXED_WIND_INIT,
            'pattern': tuple(pattern),
        },
        'wind_evol_params': dict(_FIXED_WIND_EVOL),
    }


def format_scenario(scenario: dict, name: str = "RANDOM") -> str:
    """Affiche un scénario sous forme lisible (pour debug)."""
    p = scenario['wind_init_params']['pattern']
    lines = [f"{name} = {{"]
    lines.append("  'wind_init_params': {")
    lines.append(f"    'base_speed': {scenario['wind_init_params']['base_speed']},")
    lines.append(f"    'base_max_rotation_angle_degree': "
                 f"{scenario['wind_init_params']['base_max_rotation_angle_degree']},")
    lines.append("    'pattern': (")
    for row in p:
        lines.append(f"      {row},")
    lines.append("    )")
    lines.append("  },")
    lines.append(f"  'wind_evol_params': {scenario['wind_evol_params']}")
    lines.append("}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES POIDS EXISTANTS
# ═══════════════════════════════════════════════════════════════════════════════

def load_weights_into_model(model: nn.Module, npz_path: str) -> None:
    """
    Charge les poids depuis un fichier .npz (format _save_weights) dans un
    modèle torch existant.

    Ignore les clés commençant par '_' (métriques JSON, etc.).
    """
    data    = np.load(npz_path, allow_pickle=True)
    weights = {k: data[k] for k in data.files if not k.startswith('_')}

    state_dict = {}
    for k, v in weights.items():
        state_dict[k] = torch.tensor(v)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing:
        print(f"  ⚠ Clés manquantes : {missing}")
    if unexpected:
        print(f"  ⚠ Clés inattendues : {unexpected}")
    print(f"  ✓ Poids chargés depuis {npz_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  GESTIONNAIRE D'ENVIRONNEMENTS AVEC SCÉNARIOS ROTATIFS
# ═══════════════════════════════════════════════════════════════════════════════

class ScenarioPool:
    """
    Gère un pool d'environnements SailingEnv avec renouvellement automatique
    du scénario à chaque reset.

    Paramètres
    ----------
    n_envs       : nombre total d'environnements parallèles
    orig_ratio   : fraction d'envs qui tournent sur les 3 scénarios originaux
                   (pour éviter le catastrophic forgetting)
    rng          : générateur numpy pour la reproductibilité
    SailingEnv   : classe importée de env_sailing
    get_scenario : fonction importée de wind_scenarios
    """

    def __init__(self, n_envs, orig_ratio, rng, SailingEnv, get_scenario):
        self.n_envs      = n_envs
        self.orig_ratio  = orig_ratio
        self.rng         = rng
        self.SailingEnv  = SailingEnv
        self.get_scenario = get_scenario

        self.n_orig   = max(1, int(round(n_envs * orig_ratio)))
        self.n_random = n_envs - self.n_orig

        self.envs     = []
        self.raw_obs  = []

        # Création initiale
        for i in range(n_envs):
            env, obs = self._new_env(i)
            self.envs.append(env)
            self.raw_obs.append(obs)

        print(f"[ScenarioPool] {n_envs} envs : "
              f"{self.n_orig} originaux + {self.n_random} aléatoires")

    def _new_env(self, env_idx: int):
        """Crée un nouvel env avec un scénario adapté à l'index."""
        if env_idx < self.n_orig:
            # Scénario original (cycling sur les 3)
            sc_name = ORIGINAL_SCENARIOS[env_idx % len(ORIGINAL_SCENARIOS)]
            params  = self.get_scenario(sc_name)
        else:
            # Scénario aléatoire
            params = generate_random_scenario(self.rng)

        env = self.SailingEnv(**params)
        env.seed(int(self.rng.integers(0, 2**31)))
        obs, _ = env.reset()
        return env, obs

    def reset_env(self, env_idx: int):
        """
        Réinitialise l'env `env_idx` avec un NOUVEAU scénario aléatoire
        (ou original si env_idx < n_orig).

        Appelé à chaque fin d'épisode pour maximiser la diversité.
        """
        if env_idx < self.n_orig:
            # Les envs originaux gardent leur scénario mais reset leur état
            obs, _ = self.envs[env_idx].reset()
            self.raw_obs[env_idx] = obs
        else:
            # Les envs aléatoires tirent un nouveau scénario à chaque épisode
            env, obs = self._new_env(env_idx)
            self.envs[env_idx] = env
            self.raw_obs[env_idx] = obs

        return self.raw_obs[env_idx]


# ═══════════════════════════════════════════════════════════════════════════════
#  FINE-TUNING PPO
# ═══════════════════════════════════════════════════════════════════════════════

def finetune(
    weights_in:           str   = "ppo_bc.npz",
    weights_out:          str   = "ppo_bc_finetuned.npz",
    n_envs:               int   = 20,
    total_steps:          int   = 2_000_000,
    rollout_steps:        int   = 512,
    n_epochs:             int   = 4,
    minibatch_size:       int   = 256,
    lr:                   float = 3e-5,
    gamma:                float = 0.995,
    gae_lambda:           float = 0.95,
    clip_coef:            float = 0.15,      # légèrement réduit vs training initial
    vf_coef:              float = 0.5,
    ent_coef:             float = 0.005,     # réduit : préserver la politique acquise
    max_grad_norm:        float = 0.5,
    orig_ratio:           float = 0.2,
    log_interval:         int   = 100,
    checkpoint_interval:  int   = 200,
    device_str:           str   = "auto",
    seed:                 int   = 42,
    lr_anneal:            bool  = False,     # pas d'annealing par défaut en finetune
):
    """
    Fine-tune le modèle PPO-BC sur des scénarios de vent aléatoires.

    Parameters
    ----------
    weights_in   : chemin vers le .npz à charger (poids initiaux)
    weights_out  : chemin de sauvegarde des poids fine-tunés
    orig_ratio   : fraction des envs sur les scénarios originaux [0,1]
    lr_anneal    : si True, anneal lr linéaire comme en training initial
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch requis pour le fine-tuning.")

    # ── Imports env ───────────────────────────────────────────────────────
    try:
        from src.env_sailing import SailingEnv
        from src.wind_scenarios import get_wind_scenario
    except ImportError:
        from env_sailing import SailingEnv
        from wind_scenarios import get_wind_scenario

    # ── Reproductibilité ──────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available())
        else ("cpu" if device_str == "auto" else device_str)
    )
    print(f"[FineTune] Device : {device}")
    print(f"[FineTune] Poids d'entrée  : {weights_in}")
    print(f"[FineTune] Poids de sortie : {weights_out}")
    print(f"[FineTune] Steps : {total_steps:,} | n_envs : {n_envs} | "
          f"orig_ratio : {orig_ratio:.0%} | lr : {lr:.2e}")

    # Afficher quelques exemples de scénarios générés
    print("\n[FineTune] Exemples de scénarios aléatoires :")
    demo_rng = np.random.default_rng(0)
    for i in range(2):
        sc = generate_random_scenario(demo_rng)
        print(format_scenario(sc, f"RANDOM_{i+1}"))
        print()

    # ── Modèle ────────────────────────────────────────────────────────────
    model     = _make_network(nn).to(device)
    load_weights_into_model(model, weights_in)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # ── Pool d'environnements ─────────────────────────────────────────────
    pool = ScenarioPool(n_envs, orig_ratio, rng, SailingEnv, get_wind_scenario)

    # ── Buffers ───────────────────────────────────────────────────────────
    buf_obs      = torch.zeros(rollout_steps, n_envs, N_FEATURES).to(device)
    buf_actions  = torch.zeros(rollout_steps, n_envs, dtype=torch.long).to(device)
    buf_logprobs = torch.zeros(rollout_steps, n_envs).to(device)
    buf_rewards  = torch.zeros(rollout_steps, n_envs).to(device)
    buf_dones    = torch.zeros(rollout_steps, n_envs).to(device)
    buf_values   = torch.zeros(rollout_steps, n_envs).to(device)

    next_obs = torch.tensor(
        np.stack([extract_features(o) for o in pool.raw_obs]),
        dtype=torch.float32,
    ).to(device)
    next_dones = torch.zeros(n_envs).to(device)

    # ── Métriques ─────────────────────────────────────────────────────────
    metrics = {
        'success_rate':       [],
        'collision_rate':     [],
        'mean_score':         [],
        'mean_steps_success': [],
        'mean_shaped_reward': [],
        'policy_loss':        [],
        'value_loss':         [],
        'entropy':            [],
        'approx_kl':          [],
        'n_random_scenarios': [],   # nb de scénarios aléatoires générés total
    }

    ep_rewards    = [0.0]  * n_envs
    ep_lengths    = [0]    * n_envs
    ep_successes  = [False]* n_envs
    ep_collisions = [False]* n_envs
    ep_shaped     = [0.0]  * n_envs

    recent_success   = deque(maxlen=200)
    recent_collision = deque(maxlen=200)
    recent_reward    = deque(maxlen=200)
    recent_length    = deque(maxlen=200)
    recent_shaped    = deque(maxlen=200)

    completed_episodes  = 0
    n_random_generated  = 0
    global_step         = 0
    n_updates           = total_steps // (n_envs * rollout_steps)
    t_start             = time.time()

    print(f"\n[FineTune] Début : {total_steps:,} steps → {n_updates} updates")

    for update in range(1, n_updates + 1):

        # LR optionnel
        if lr_anneal:
            lr_now = lr * (1.0 - (update - 1) / n_updates)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_now
        else:
            lr_now = lr

        # ── Collecte du rollout ───────────────────────────────────────────
        for step in range(rollout_steps):
            global_step += n_envs
            buf_obs[step]   = next_obs
            buf_dones[step] = next_dones

            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(next_obs)
            buf_actions[step]  = action
            buf_logprobs[step] = log_prob
            buf_values[step]   = value

            new_obs_list  = []
            step_rewards  = []
            step_dones    = []

            for i in range(n_envs):
                env   = pool.envs[i]
                a     = action[i].item()
                o_raw = pool.raw_obs[i]

                o_next, r, terminated, truncated, info = env.step(a)

                sr, _ = shaped_reward(
                    o_raw, o_next, r, terminated,
                    info.get('is_stuck', False), ep_lengths[i],
                )

                ep_rewards[i]   += r
                ep_lengths[i]   += 1
                ep_shaped[i]    += sr
                if info.get('is_stuck', False):
                    ep_collisions[i] = True
                if r > 50:
                    ep_successes[i] = True

                done = terminated or truncated
                step_rewards.append(sr)
                step_dones.append(float(done))
                pool.raw_obs[i] = o_next

                if done:
                    recent_success.append(float(ep_successes[i]))
                    recent_collision.append(float(ep_collisions[i]))
                    recent_reward.append(ep_rewards[i])
                    recent_shaped.append(ep_shaped[i])
                    if ep_successes[i]:
                        recent_length.append(ep_lengths[i])

                    completed_episodes += 1

                    # Log
                    if completed_episodes % log_interval == 0:
                        elapsed = time.time() - t_start
                        print(
                            f"[Ep {completed_episodes:6d} | Step {global_step:8d} | "
                            f"{elapsed/60:.1f}min] "
                            f"Succès={np.mean(recent_success)*100:.1f}% | "
                            f"Collision={np.mean(recent_collision)*100:.1f}% | "
                            f"Score={np.mean(recent_reward):.2f} | "
                            f"Steps(succ)={np.mean(recent_length) if recent_length else 0:.1f} | "
                            f"ShapedR={np.mean(recent_shaped):.2f} | "
                            f"Scénarios générés={n_random_generated} | "
                            f"lr={lr_now:.2e}"
                        )
                        metrics['success_rate'].append(float(np.mean(recent_success)))
                        metrics['collision_rate'].append(float(np.mean(recent_collision)))
                        metrics['mean_score'].append(float(np.mean(recent_reward)))
                        metrics['mean_shaped_reward'].append(float(np.mean(recent_shaped)))
                        metrics['n_random_scenarios'].append(n_random_generated)
                        if recent_length:
                            metrics['mean_steps_success'].append(float(np.mean(recent_length)))

                    # Checkpoint
                    if completed_episodes % checkpoint_interval == 0:
                        _save_weights(model, weights_out, metrics)
                        print(f"  ✓ Checkpoint → {weights_out}")

                    # Reset avec nouveau scénario aléatoire
                    new_o = pool.reset_env(i)
                    if i >= pool.n_orig:
                        n_random_generated += 1
                    o_next = new_o

                    ep_rewards[i]    = 0.0
                    ep_lengths[i]    = 0
                    ep_successes[i]  = False
                    ep_collisions[i] = False
                    ep_shaped[i]     = 0.0

                new_obs_list.append(extract_features(o_next))

            buf_rewards[step] = torch.tensor(step_rewards, dtype=torch.float32).to(device)
            next_obs   = torch.tensor(np.stack(new_obs_list), dtype=torch.float32).to(device)
            next_dones = torch.tensor(step_dones,             dtype=torch.float32).to(device)

        # ── GAE ───────────────────────────────────────────────────────────
        with torch.no_grad():
            next_value = model.get_action_and_value(next_obs)[3]
            advantages = torch.zeros_like(buf_rewards)
            last_gae   = 0.0
            for t in reversed(range(rollout_steps)):
                if t == rollout_steps - 1:
                    nnt = 1.0 - next_dones
                    nv  = next_value
                else:
                    nnt = 1.0 - buf_dones[t + 1]
                    nv  = buf_values[t + 1]
                delta    = buf_rewards[t] + gamma * nv * nnt - buf_values[t]
                last_gae = delta + gamma * gae_lambda * nnt * last_gae
                advantages[t] = last_gae
            returns = advantages + buf_values

        # ── Mise à jour PPO ───────────────────────────────────────────────
        b_obs      = buf_obs.reshape(-1, N_FEATURES)
        b_actions  = buf_actions.reshape(-1)
        b_logprobs = buf_logprobs.reshape(-1)
        b_advs     = (advantages.reshape(-1))
        b_returns  = returns.reshape(-1)
        b_advs     = (b_advs - b_advs.mean()) / (b_advs.std() + 1e-8)

        n_samples = rollout_steps * n_envs
        inds      = np.arange(n_samples)
        upg, uvf, uent, ukl = [], [], [], []

        for epoch in range(n_epochs):
            np.random.shuffle(inds)
            for start in range(0, n_samples, minibatch_size):
                mb = inds[start:start + minibatch_size]

                _, new_lp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb], b_actions[mb]
                )
                log_ratio  = new_lp - b_logprobs[mb]
                ratio      = log_ratio.exp()
                approx_kl  = ((ratio - 1) - log_ratio).mean().item()
                mb_adv     = b_advs[mb]

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
                ).mean()
                vf_loss  = 0.5 * (new_val - b_returns[mb]).pow(2).mean()
                ent_loss = entropy.mean()
                loss     = pg_loss + vf_coef * vf_loss - ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                upg.append(pg_loss.item())
                uvf.append(vf_loss.item())
                uent.append(ent_loss.item())
                ukl.append(approx_kl)

        metrics['policy_loss'].append(float(np.mean(upg)))
        metrics['value_loss'].append(float(np.mean(uvf)))
        metrics['entropy'].append(float(np.mean(uent)))
        metrics['approx_kl'].append(float(np.mean(ukl)))

    # ── Sauvegarde finale ─────────────────────────────────────────────────
    _save_weights(model, weights_out, metrics)
    elapsed = time.time() - t_start
    print(f"\n[FineTune] Terminé en {elapsed/60:.1f} min")
    print(f"  Scénarios aléatoires générés : {n_random_generated:,}")
    print(f"  Taux de succès final  : {np.mean(list(recent_success))*100:.1f}%")
    print(f"  Taux de collision fin : {np.mean(list(recent_collision))*100:.1f}%")
    print(f"  Score moyen final     : {np.mean(list(recent_reward)):.3f}")
    print(f"  Poids sauvegardés     : {weights_out}")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning PPO-BC sur scénarios de vent aléatoires",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights_in",  type=str,   default="ppo_bc.npz",
                        help="Fichier .npz des poids à charger")
    parser.add_argument("--weights_out", type=str,   default="ppo_bc_finetuned.npz",
                        help="Fichier .npz de sauvegarde des poids fine-tunés")
    parser.add_argument("--steps",       type=int,   default=2_000_000,
                        help="Total steps de fine-tuning")
    parser.add_argument("--n_envs",      type=int,   default=20,
                        help="Nombre d'environnements parallèles")
    parser.add_argument("--rollout",     type=int,   default=512,
                        help="Steps par rollout")
    parser.add_argument("--lr",          type=float, default=3e-5,
                        help="Learning rate (doit être ~10x plus petit que l'init)")
    parser.add_argument("--ent_coef",    type=float, default=0.005,
                        help="Coefficient d'entropie (plus petit qu'en training initial)")
    parser.add_argument("--clip_coef",   type=float, default=0.15,
                        help="Clip PPO (légèrement réduit pour stabilité)")
    parser.add_argument("--orig_ratio",  type=float, default=0.2,
                        help="Fraction des envs sur scénarios originaux [0,1]")
    parser.add_argument("--log_interval",type=int,   default=100,
                        help="Log toutes les N épisodes")
    parser.add_argument("--checkpoint",  type=int,   default=200,
                        help="Checkpoint toutes les N épisodes")
    parser.add_argument("--device",      type=str,   default="auto",
                        help="Device torch : auto / cpu / cuda")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Graine aléatoire globale")
    parser.add_argument("--lr_anneal",   action="store_true",
                        help="Anneal lr linéairement (non recommandé en finetune)")

    args = parser.parse_args()

    print("=" * 65)
    print("FINE-TUNING PPO-BC — Scénarios aléatoires")
    print("=" * 65)

    metrics = finetune(
        weights_in           = args.weights_in,
        weights_out          = args.weights_out,
        n_envs               = args.n_envs,
        total_steps          = args.steps,
        rollout_steps        = args.rollout,
        lr                   = args.lr,
        ent_coef             = args.ent_coef,
        clip_coef            = args.clip_coef,
        orig_ratio           = args.orig_ratio,
        log_interval         = args.log_interval,
        checkpoint_interval  = args.checkpoint,
        device_str           = args.device,
        seed                 = args.seed,
        lr_anneal            = args.lr_anneal,
    )

    metrics_path = args.weights_out.replace(".npz", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMétriques sauvegardées → {metrics_path}")