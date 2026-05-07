from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np


from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario

# --- Configuration ---
NUM_EPISODES = 1000
GAMMA = 0.995
LR = 1e-4
UPDATE_TIMESTEP = 2000 
SCENARIOS = ['training_1', 'training_2', 'training_3']
CROP_SIZE = 32

# --- Configuration du Curriculum ---
# On commence à introduire l'aléa après 1/3 de l'entraînement
RANDOM_START_EP = NUM_EPISODES // 3  
# On introduit la vitesse initiale après 2/3
RANDOM_VELO_EP = (2 * NUM_EPISODES) // 3 

class SailingNet(nn.Module):
    """Architecture hybride CNN + MLP pour le traitement des cartes et vecteurs."""
    def __init__(self, crop_size=32, n_actions=9):
        super().__init__()
        # Branche spatiale (Local Crop)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        # Branche vectorielle (Vitesse, But, Vent)
        self.mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU())
        
        # Calcul de la taille de sortie CNN (ici 32 filtres * 16x16 pixels après MaxPool)
        cnn_out_size = 32 * (crop_size // 2) * (crop_size // 2)
        
        self.actor = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 128), nn.ReLU(),
            nn.Linear(128, n_actions), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(cnn_out_size + 32, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, local_map, scalars):
        x_cnn = self.cnn(local_map)
        x_mlp = self.mlp(scalars)
        combined = torch.cat([x_cnn, x_mlp], dim=1)
        return self.actor(combined), self.critic(combined)

class Memory:
    """Buffer pour stocker les trajectoires avant l'update PPO."""
    def __init__(self):
        self.actions, self.states_map, self.states_scalars = [], [], []
        self.logprobs, self.rewards, self.is_terminals = [], [], []

    def clear(self):
        self.actions.clear(); self.states_map.clear(); self.states_scalars.clear()
        self.logprobs.clear(); self.rewards.clear(); self.is_terminals.clear()

class PPOSailingAgent:
    def __init__(self, crop_size=32):
        self.crop_size = crop_size
        self.policy = SailingNet(crop_size=crop_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        # Décroissance du LR sur la durée totale
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_EPISODES)
        
        self.policy_old = SailingNet(crop_size=crop_size)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.loss_history = {'actor': [], 'critic': []}

    def get_local_crop(self, obs):
        """Extrait une fenêtre 3x32x32 autour du bateau (Terre + Vent U/V)."""
        pos = obs[0:2].astype(int)
        wmap = obs[32774:49158].reshape(128, 128)
        wfield = obs[6:32774].reshape(128, 128, 2)
        pad = self.crop_size // 2
        
        # Padding constant pour les bords de map
        wmap_p = np.pad(wmap, pad, constant_values=1)
        wf_p = np.pad(wfield, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)
        
        y, x = pos[1] + pad, pos[0] + pad
        crop = np.zeros((3, self.crop_size, self.crop_size))
        crop[0] = wmap_p[y-pad:y+pad, x-pad:x+pad]
        crop[1:] = wf_p[y-pad:y+pad, x-pad:x+pad].transpose(2, 0, 1)
        return torch.FloatTensor(crop).unsqueeze(0)

    def select_action(self, obs, goal, prev_angle):
        """Calcule les inputs et sélectionne une action selon la vieille politique."""
        with torch.no_grad():
            m_input = self.get_local_crop(obs)
            curr_w = obs[4:6]
            curr_a = np.arctan2(curr_w[1], curr_w[0])
            da = curr_a - prev_angle if prev_angle else 0
            
            s_input = torch.FloatTensor([obs[2], obs[3], goal[0]-obs[0], goal[1]-obs[1], da]).unsqueeze(0)
            
            probs, _ = self.policy_old(m_input, s_input)
            dist = Categorical(probs)
            action = dist.sample()
            
            return action.item(), dist.log_prob(action), curr_a, m_input, s_input

    def update(self, memory):
        # Calcul des Returns normalisés
        rewards = []
        discounted_r = 0
        for r, term in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if term: discounted_r = 0
            discounted_r = r + (GAMMA * discounted_r)
            rewards.insert(0, discounted_r)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Conversion buffer
        map_s = torch.cat(memory.states_map); sca_s = torch.cat(memory.states_scalars)
        act_s = torch.tensor(memory.actions); lp_s = torch.stack(memory.logprobs).detach()

        for _ in range(5): # K-epochs
            probs, vals = self.policy(map_s, sca_s)
            dist = Categorical(probs)
            
            ratios = torch.exp(dist.log_prob(act_s) - lp_s)
            adv = rewards - vals.detach().squeeze()
            
            surr1 = ratios * adv
            surr2 = torch.clamp(ratios, 0.8, 1.2) * adv
            
            a_loss = -torch.min(surr1, surr2).mean()
            c_loss = 0.5 * F.mse_loss(vals.squeeze(), rewards)
            
            self.optimizer.zero_grad()
            (a_loss + c_loss - 0.01 * dist.entropy().mean()).backward()
            self.optimizer.step()
            
            self.loss_history['actor'].append(a_loss.item())
        
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        weights = {n: p.detach().cpu().numpy() for n, p in self.policy.named_parameters()}
        np.savez(path.replace(".pth", ".npz"), **weights)

# --- Boucle Principale ---
agent = PPOSailingAgent(CROP_SIZE)
memory = Memory()
history = {'rewards': [], 'success': [], 'steps': []}
scenario_stats = {s: {'rewards': [], 'success': [], 'steps': []} for s in SCENARIOS}
total_step = 0

for ep in tqdm(range(NUM_EPISODES)):
    scenario = SCENARIOS[ep % 3]
    env = SailingEnv(**get_wind_scenario(scenario))
    
    # --- Gestion dynamique des options de Reset (Curriculum) ---
    options = {}
    if ep > RANDOM_START_EP:
        rx = int(np.random.randint(5, 123)) 
        ry = int(np.random.randint(5, 123))
        options["random_start"] = True
        options["start_position"] = [rx, ry] 
        options["wind_start_steps"] = int(np.random.randint(0, 100))
        ep_steps = options["wind_start_steps"]

    if ep > RANDOM_VELO_EP:
        options["random_velocity"] = True
        
    obs, info = env.reset(options=options)
    goal = env.goal_position
    
    # CRITIQUE : Recalculer la distance après le reset (surtout si random_start)
    p_angle = None
    p_dist = np.linalg.norm(info['position'] - goal)
    ep_r = 0
    ep_steps = 0

    for t in range(500):
        total_step += 1
        ep_steps += 1
        
        # Action & Store
        act, lp, p_angle, m_in, s_in = agent.select_action(obs, goal, p_angle)
        next_obs, r, done, trunc, info = env.step(act)
        
        # --- Reward Shaping avec Border Penalty ---
        speed = np.linalg.norm(next_obs[2:4]) # Vitesse actuelle
        curr_dist = np.linalg.norm(info['position'] - goal)
        x, y = info['position']
        
        # 1. Progression vers le but
        shaped_r = r + (p_dist - curr_dist) * 0.1 
        
        # 2. Malus Stagnation (speed < 0.1)
        if speed < 0.1: shaped_r -= 0.5
        
        # 3. Malus Bordure (si à moins de 5 unités du bord 128x128)
        if x < 5 or x > 123 or y < 5 or y > 123:
            shaped_r -= 1.0
            
        # 4. Collision
        if info.get('is_stuck', False): shaped_r = -20.0
        
        memory.states_map.append(m_in); memory.states_scalars.append(s_in)
        memory.actions.append(act); memory.logprobs.append(lp)
        memory.rewards.append(shaped_r); memory.is_terminals.append(done or trunc)

        obs, ep_r, p_dist = next_obs, ep_r + r, curr_dist

        if total_step % UPDATE_TIMESTEP == 0:
            agent.update(memory); memory.clear()
        if done or trunc: break
    
    # --- Logs & Stats ---
    agent.scheduler.step()
    
    # On ne logue les steps que si l'épisode est un succès (done et pas trunc)
    actual_steps = ep_steps if done else 500 
    
    history['rewards'].append(ep_r)
    history['success'].append(1 if done else 0)
    history['steps'].append(actual_steps)
    
    scenario_stats[scenario]['rewards'].append(ep_r)
    scenario_stats[scenario]['success'].append(1 if done else 0)
    scenario_stats[scenario]['steps'].append(actual_steps)

    # --- Affichage périodique tous les 30 épisodes ---
    if (ep + 1) % 30 == 0:
        curr_lr = agent.optimizer.param_groups[0]['lr']
        print(f"\n--- Bilan Eps {ep-28}-{ep+1} | LR: {curr_lr:.2e} ---")
        
        for s in SCENARIOS:
            s_data = scenario_stats[s]
            sr = np.mean(s_data['success']) * 100
            rw = np.mean(s_data['rewards'])
            # Moyenne des steps uniquement sur les réussites pour voir la qualité
            success_steps = [st for st, succ in zip(s_data['steps'], s_data['success']) if succ == 1]
            avg_step = np.mean(success_steps) if success_steps else 500
            
            indicator = "⭐" if avg_step < 60 else "📈" if avg_step < 90 else "🐌"
            print(f"[{s:10s}] Success: {sr:3.0f}% | Reward: {rw:5.1f} | Steps: {avg_step:4.1f} {indicator}")
            
        # Reset des stats locales
        scenario_stats = {s: {'rewards': [], 'success': [], 'steps': []} for s in SCENARIOS}

agent.save("alea_model.pth")
np.savez("history.npz", **history)