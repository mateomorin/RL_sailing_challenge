import numpy as np
import sys
import os

try:
    from base_agent import BaseAgent
except ImportError:
    try:
        from agents.base_agent import BaseAgent
    except ImportError:
        from src.agents.base_agent import BaseAgent

# from evaluator.base_agent import BaseAgent


def calculate_sailing_efficiency(boat_direction, wind_direction):
    """
    Calculate sailing efficiency based on the angle between boat direction and wind.
    
    Args:
        boat_direction: Normalized vector of boat's desired direction
        wind_direction: Normalized vector of wind direction (where wind is going TO)
        
    Returns:
        sailing_efficiency: Float between 0.05 and 1.0 representing how efficiently the boat can sail
    """
    # Invert wind direction to get where wind is coming FROM
    wind_from = -wind_direction
    
    # Calculate angle between wind and direction
    wind_angle = np.arccos(np.clip(
        np.dot(wind_from, boat_direction), -1.0, 1.0))
    
    # Calculate sailing efficiency based on angle to wind
    if wind_angle < np.pi/4:  # Less than 45 degrees to wind
        sailing_efficiency = 0.05  # Small but non-zero efficiency in no-go zone
    elif wind_angle < np.pi/2:  # Between 45 and 90 degrees
        sailing_efficiency = 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)  # Linear increase to 1.0
    elif wind_angle < 3*np.pi/4:  # Between 90 and 135 degrees
        sailing_efficiency = 1.0  # Maximum efficiency
    else:  # More than 135 degrees
        sailing_efficiency = 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4)  # Linear decrease
        sailing_efficiency = max(0.5, sailing_efficiency)  # But still decent
    
    return sailing_efficiency 


class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.grid_size = (128, 128)
        self.goal_pos = np.array([64, 127])
        self.world_map = None
        
        # Constantes de l'environnement
        self.boat_performance = 0.4 
        self.max_speed = 8.0
        self.inertia_factor = 0.3
        
        self.directions = [
            np.array([0, 1]),   np.array([1, 1]),   np.array([1, 0]),
            np.array([1, -1]),  np.array([0, -1]),  np.array([-1, -1]),
            np.array([-1, 0]),  np.array([-1, 1]),  np.array([0, 0])
        ]
    
    def _get_efficiency(self, boat_dir, wind_vec):
        """Ta fonction de rendement selon l'angle du vent."""
        if np.all(boat_dir == 0): return 0.05
        boat_dir_n = boat_dir / (np.linalg.norm(boat_dir) + 1e-6)
        wind_norm = np.linalg.norm(wind_vec)
        if wind_norm < 1e-6: return 1.0
        
        wind_from = -(wind_vec / wind_norm)
        wind_angle = np.arccos(np.clip(np.dot(wind_from, boat_dir_n), -1.0, 1.0))
        
        if wind_angle < np.pi/4: return 0.05
        elif wind_angle < np.pi/2: return 0.5 + 0.5 * (wind_angle - np.pi/4) / (np.pi/4)
        elif wind_angle < 3*np.pi/4: return 1.0
        else: return max(0.5, 1.0 - 0.5 * (wind_angle - 3*np.pi/4) / (np.pi/4))

    def _predict_velocity(self, current_vel, wind, action_dir):
        """Réplique exacte de la physique du simulateur."""
        wind_norm = np.linalg.norm(wind)
        if wind_norm <= 0:
            return (current_vel * self.inertia_factor).astype(np.int32)

        eff = calculate_sailing_efficiency(action_dir, wind / wind_norm)

        # theoretical_velocity
        theo_vel = action_dir * eff * wind_norm * self.boat_performance
        
        # Max speed limit
        theo_speed = np.linalg.norm(theo_vel)
        if theo_speed > self.max_speed:
            theo_vel = (theo_vel / theo_speed) * self.max_speed

        # Inertie
        new_vel = theo_vel + self.inertia_factor * (current_vel - theo_vel)

        # Final speed limit
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = (new_vel / speed) * self.max_speed

        # Discrétisation
        v_final = np.where(new_vel < 0, np.ceil(new_vel), np.floor(new_vel)).astype(np.int32)
        return v_final

    def act(self, observation: np.ndarray) -> int:
        pos = observation[0:2]
        vel = observation[2:4]
        wind_at_pos = observation[4:6]
        
        if self.world_map is None:
            self.world_map = observation[32774:49158].reshape((128, 128))

        # Calcul de la distance à l'objectif
        to_goal = self.goal_pos - pos
        dist_to_goal = np.linalg.norm(to_goal)
        target_dir = to_goal / (dist_to_goal + 1e-6)

        # --- DETECTION MODE APPROCHE FINALE ---
        # Si on est à moins de 5 cases, on devient très prudent
        is_final_approach = dist_to_goal < 5.0 

        best_action = 8
        best_score = -float('inf')

        for i in range(9):
            action_dir = self.directions[i]
            pred_vel = self._predict_velocity(vel, wind_at_pos, action_dir)
            future_pos = pos + pred_vel
            
            # 1. Sécurité stricte (Sortie de cadre ou Terre)
            if not (0 <= future_pos[0] < 128 and 0 <= future_pos[1] < 128):
                continue
            if self.world_map[int(round(future_pos[1]))][int(round(future_pos[0]))] == 1:
                continue

            # 2. Calcul du score
            if is_final_approach:
                # En approche finale, on veut :
                # - Réduire la distance le plus possible (Précision)
                # - Ne pas avoir une vitesse trop grande (Contrôle)
                new_dist = np.linalg.norm(self.goal_pos - future_pos)
                future_speed = np.linalg.norm(pred_vel)
                
                # On veut minimiser la distance, donc maximiser (-distance)
                # On ajoute un malus à la vitesse pour éviter de "sauter" par dessus le but
                score = -new_dist - (future_speed * 0.1)
                
                # Si l'action nous met pile sur le but (dist < 1.5), c'est l'action parfaite
                if new_dist < 1.5:
                    score += 1000
            else:
                # Mode normal : VMG optimisé
                vmg = np.dot(pred_vel.astype(float), target_dir)
                
                # Marge de sécurité terre
                safety_margin = 1.0
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = int(round(future_pos[0]+dx)), int(round(future_pos[1]+dy))
                    if 0 <= nx < 128 and 0 <= ny < 128 and self.world_map[ny][nx] == 1:
                        safety_margin = 0.2
                
                score = vmg * safety_margin

            if score > best_score:
                best_score = score
                best_action = i

        return best_action

    def reset(self):
        self.world_map = None