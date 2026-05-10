from src.agents.test import ScenarioDecoder, KNOWN_SCENARIOS, _simulate_probe_steps
import src.env_sailing as env_sailing

# Ajouts à KNOWN_SCENARIOS
EXTENDED_SCENARIOS = {
    'training_4_vortex': {
        'wind_init_params': {
            'base_speed': 12.0,
            'base_max_rotation_angle_degree': 15,
            'pattern': [
                [(1,0), (1,1), (0,1)],
                [(-1,1), (0,0), (1,-1)],
                [(0,-1), (-1,-1), (-1,0)],
            ],
        },
        'wind_evol_params': {'mean_rotation_angle_degree': 1.0, 'std_rotation_angle_degree': 0.5},
    },
    'training_5_north_strong': {
        'wind_init_params': {
            'base_speed': 15.0,
            'base_max_rotation_angle_degree': 5,
            'pattern': [
                [(0,1), (0,1), (0,1)],
                [(0,1), (0,1), (0,1)],
                [(0,1), (0,1), (0,1)],
            ],
        },
        'wind_evol_params': {'mean_rotation_angle_degree': 5.0, 'std_rotation_angle_degree': 2.0},
    }
}
KNOWN_SCENARIOS.update(EXTENDED_SCENARIOS)


def test_decoder_accuracy():
    print("=== TEST DU SYSTEME DE DECODAGE ===")
    
    # 1. Initialisation du décodeur
    decoder = ScenarioDecoder(force_recompute=True)
    
    # 2. Pré-calcul de la base de données (Le "Cerveau" de l'agent)
    # On simule les empreintes pour TOUS les scénarios connus sur les seeds 1 à 10
    test_seeds = list(range(1, 11))
    decoder.SEEDS = test_seeds
    decoder.precompute(KNOWN_SCENARIOS)
    
    # 3. Simulation du "Scénario Mystère"
    # On choisit par exemple 'training_2' comme étant le scénario réel du test
    target_scenario_name = 'training_2'
    target_cfg = KNOWN_SCENARIOS[target_scenario_name]
    
    print(f"\n--- Simulation du scénario mystère : {target_scenario_name} ---")
    
    # On récupère les steps que le ProbeAgent ferait réellement sur Codabench
    observed_results = _simulate_probe_steps(target_cfg, seeds=test_seeds)
    
    # 4. Décodage
    print("\n--- Tentative de décodage ---")
    prediction = decoder.decode_from_logs(observed_results)
    
    # 5. Validation
    if prediction['scenario'] == target_scenario_name:
        print(f"\n✅ SUCCÈS : Scénario '{prediction['scenario']}' identifié avec "
              f"{prediction['confidence']*100:.1f}% de confiance.")
    else:
        print(f"\n❌ ÉCHEC : Le décodeur a prédit '{prediction['scenario']}' "
              f"au lieu de '{target_scenario_name}'.")

if __name__ == "__main__":
    # Assure-toi que SailingEnv est importable ou simule-le ici
    test_decoder_accuracy()