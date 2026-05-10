import numpy as np
import json
import time

# ============================================================
# CONFIGURATION DU POINT DE DÉPART (Ton dernier meilleur)
# ============================================================
BASE_PATTERN = np.array([
                [
                    (
                        0.3901069886585547,
                        -0.9207695354429111
                    ),
                    (
                        0.7303332675538328,
                        0.6830910029448066
                    ),
                    (
                        0.8836187279364603,
                        -0.46820715889438475
                    )
                ],
                [
                    (
                        0.20215188256028796,
                        0.9793541833153783
                    ),
                    (
                        0.9963275557953793,
                        0.08562360400500091
                    ),
                    (
                        -0.6034649664978712,
                        -0.7973895122270691
                    )
                ],
                [
                    (
                        0.9381347465686567,
                        0.34627041063388897
                    ),
                    (
                        0.8435431072671562,
                        0.5370614733734593
                    ),
                    (
                        0.7082873561641528,
                        0.7059242318393629
                    )
                ]
            ])

CURRENT_BEST_SCENARIO = {
    'wind_init_params': {
        'base_speed': 10.0,
        'base_max_rotation_angle_degree': 10,
        'pattern': BASE_PATTERN.tolist(),
    },
    'wind_evol_params': {
        'mean_rotation_angle_degree': 3.0,
        'std_rotation_angle_degree': 0.8,
    }
}

# Charger les steps observés
OBSERVED_STEPS = {
    1:50, 2:49, 3:56, 4:60, 5:57, 6:55, 7:45, 8:49, 9:49, 10:46,
    11:48, 12:48, 13:54, 14:45, 15:49, 16:60, 17:59, 18:49, 19:48, 20:45,
    21:48, 22:49, 23:53, 24:57, 25:48, 26:48, 27:49, 28:49, 29:49, 30:51,
    31:47, 32:56, 33:50, 34:49, 35:56, 36:57, 37:61, 38:51, 39:58, 40:53,
    41:48, 42:48, 43:55, 44:59, 45:53, 46:51, 47:56, 48:47, 49:50, 50:54,
}
OBS_VEC = np.array([OBSERVED_STEPS[s] for s in range(1, 51)])

# ============================================================
# FONCTIONS D'EXPLORATION
# ============================================================

def mutate_pattern(base_pat, intensity=0.05):
    """Applique une petite variation aléatoire sur le pattern."""
    noise = np.random.normal(0, intensity, base_pat.shape)
    new_pat = base_pat + noise
    # Normalisation optionnelle pour garder des vecteurs de norme proche de 1
    norms = np.linalg.norm(new_pat, axis=2, keepdims=True)
    return new_pat / norms 

def run_pattern_search(n_iterations=100, mutation_intensity=0.03):
    """Boucle d'exploration uniquement sur le pattern."""
    from src.utils.decode_scenario import simulate_scenario_steps, score # Importer tes fonctions de simu
    
    best_pat = BASE_PATTERN.copy()
    best_score = 999.0 # Initialiser haut
    
    print(f"Démarrage de l'exploration (Incrémentale) - {n_iterations} itérations")
    
    for i in range(n_iterations):
        # 1. Générer un candidat proche du meilleur actuel
        candidate_pat = mutate_pattern(best_pat, intensity=mutation_intensity)
        
        test_cfg = CURRENT_BEST_SCENARIO.copy()
        test_cfg['wind_init_params']['pattern'] = candidate_pat.tolist()
        
        # 2. Simuler
        sim_vec = simulate_scenario_steps(test_cfg)
        current_score = score(sim_vec, OBS_VEC)
        
        # 3. Si c'est mieux, on garde et on affiche
        if current_score < best_score:
            best_score = current_score
            best_pat = candidate_pat
            l2 = np.linalg.norm(sim_vec - OBS_VEC)
            corr = np.corrcoef(sim_vec, OBS_VEC)[0,1]
            
            print(f"[{i:03d}] ★ Nouveau Meilleur ! Score: {best_score:.4f} | L2: {l2:.2f} | Corr: {corr:.4f}")
            
            # Sauvegarde temporaire pour ne rien perdre
            save_result(test_cfg, best_score)

    return best_pat

def save_result(cfg, score_val):
    with open("best_pattern_found.json", "w") as f:
        json.dump({"score": score_val, "scenario": cfg}, f, indent=4)

# ============================================================
# EXECUTION
# ============================================================
if __name__ == "__main__":
    # Note: Assure-toi que simulate_scenario_steps est accessible
    # n_iterations: plus c'est haut, plus il affine.
    # mutation_intensity: 0.01 pour de l'ajustement chirurgical, 0.1 pour explorer plus large.
    final_pattern = run_pattern_search(n_iterations=200, mutation_intensity=0.05)