from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario

class SailingNet(nn.Module):
    def __init__(self, crop_size=32, n_actions=9):
        super().__init__()
        self.crop_size = crop_size
        
        # Branche CNN : Traite le crop local (3 canaux : Terre, Wind_U, Wind_V)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Branche Scalaire : [vx, vy, dx_goal, dy_goal, wind_angle_delta]
        # On calcule wind_angle_delta en comparant le vent actuel au précédent
        self.mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU()
        )
        
        # Fusion des deux branches
        # Taille CNN (32x16x16 si crop=32 et 1 pool) = 8192 (à ajuster selon crop)
        cnn_out_size = 32 * (crop_size // 2) * (crop_size // 2)
        
        self.actor = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, local_map, scalars):
        x_cnn = self.cnn(local_map)
        x_mlp = self.mlp(scalars)
        x_combined = torch.cat([x_cnn, x_mlp], dim=1)
        
        probs = self.actor(x_combined)
        value = self.critic(x_combined)
        return probs, value


class DeepSailingAgent:
    def __init__(self, crop_size=32):
        self.model = SailingNet(crop_size=crop_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0007)
        self.crop_size = crop_size
        self.prev_wind = None

    def get_local_crop(self, observation):
        pos = observation[0:2].astype(int)
        # Extraire World Map (32774:49158) et Wind Field (6:32774)
        wmap = observation[32774:49158].reshape(128, 128)
        wfield = observation[6:32774].reshape(128, 128, 2)
        
        # Padding pour gérer les bords
        pad = self.crop_size // 2
        wmap_padded = np.pad(wmap, pad, constant_values=1)
        wfield_padded = np.pad(wfield, ((pad,pad),(pad,pad),(0,0)), constant_values=0)
        
        # Crop centré sur pos (ajusté avec le padding)
        y, x = pos[1] + pad, pos[0] + pad
        crop_wmap = wmap_padded[y-pad:y+pad, x-pad:x+pad]
        crop_wfield = wfield_padded[y-pad:y+pad, x-pad:x+pad]
        
        # Concatenate: [Terre, Wind_U, Wind_V] -> shape (3, crop, crop)
        combined = np.zeros((3, self.crop_size, self.crop_size))
        combined[0] = crop_wmap
        combined[1:] = crop_wfield.transpose(2, 0, 1)
        return torch.FloatTensor(combined).unsqueeze(0)

    def train_step(self, obs, goal, prev_wind_angle):
        # 1. Prépare Inputs
        local_map = self.get_local_crop(obs)
        curr_wind = obs[4:6]
        curr_angle = np.arctan2(curr_wind[1], curr_wind[0])
        angle_delta = curr_angle - prev_wind_angle if prev_wind_angle else 0
        
        to_goal = goal - obs[0:2]
        scalars = torch.FloatTensor([obs[2], obs[3], to_goal[0], to_goal[1], angle_delta]).unsqueeze(0)
        
        # 2. Forward
        probs, value = self.model(local_map, scalars)
        return probs, value, curr_angle


# --- Initialisation ---
agent = DeepSailingAgent()
num_episodes = 2000
gamma = 0.995
np.random.seed(42)
scenarios = ['training_1', 'training_2', 'training_3']

# Historique complet pour analyse post-entraînement
history = {
    'episode_reward': [],
    'actor_loss': [],
    'critic_loss': [],
    'total_loss': [],
    'steps': [],
    'success': []
}

print(f"Starting training: {num_episodes} episodes...")

for ep in tqdm(range(num_episodes)):
    scenario = scenarios[ep % 3]
    env = SailingEnv(**get_wind_scenario(scenario))
    obs, info = env.reset()
    
    log_probs, values, rewards = [], [], []
    prev_angle = None
    prev_dist = np.linalg.norm(info['position'] - env.goal_position)
    ep_total_raw_reward = 0  # On stocke la reward brute de l'env pour le monitoring

    # --- Episode Loop ---
    for t in range(500):
        probs, val, prev_angle = agent.train_step(obs, env.goal_position, prev_angle)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        next_obs, reward, done, truncated, info = env.step(action.item())
        
        # Reward Shaping pour l'apprentissage
        curr_dist = np.linalg.norm(info['position'] - env.goal_position)
        shaped_reward = reward + (prev_dist - curr_dist) * 0.5
        if info.get('is_stuck', False): shaped_reward = -20.0
        
        log_probs.append(dist.log_prob(action))
        values.append(val)
        rewards.append(shaped_reward)
        
        ep_total_raw_reward += reward
        obs = next_obs
        prev_dist = curr_dist
        if done or truncated: break

    # --- Update Policy (A2C) ---
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.FloatTensor(returns)
    log_probs = torch.stack(log_probs)
    values = torch.cat(values).squeeze()
    
    advantage = returns - values.detach()
    a_loss = -(log_probs * advantage).mean()
    c_loss = F.mse_loss(values, returns)
    total_loss = a_loss + 0.5 * c_loss
    
    agent.optimizer.zero_grad()
    total_loss.backward()
    agent.optimizer.step()

    # --- Enregistrement des stats ---
    history['episode_reward'].append(ep_total_raw_reward)
    history['actor_loss'].append(a_loss.item())
    history['critic_loss'].append(c_loss.item())
    history['total_loss'].append(total_loss.item())
    history['steps'].append(t + 1)
    history['success'].append(1 if done else 0)

    # --- Affichage tous les 20 épisodes ---
    if (ep + 1) % 20 == 0:
        avg_rew = np.mean(history['episode_reward'][-20:])
        avg_loss = np.mean(history['total_loss'][-20:])
        success_rate = np.mean(history['success'][-20:]) * 100
        tqdm.write(f"Ep {ep+1:4d} | Reward: {avg_rew:6.1f} | Loss: {avg_loss:8.4f} | Success: {success_rate:3.0f}%")

# --- Sauvegarde finale ---
# 1. Les poids pour l'agent Numpy
weights_dict = {name: param.detach().cpu().numpy() for name, param in agent.model.named_parameters()}
np.savez("sailing_model_weights.npz", **weights_dict)

# 2. L'historique d'entraînement pour analyse (Matplotlib)
np.savez("training_history.npz", **history)

print("\nTraining Finished! Models and history saved.")