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

# Episodes
NUM_EPISODES = 5000
RANDOM_START_EP = NUM_EPISODES // 2  
RANDOM_VELO_EP = (3 * NUM_EPISODES) // 4
SCENARIOS = ['training_1', 'training_2', 'training_3']

# Modèle
GAMMA = 0.995
UPDATE_TIMESTEP = 2000 
LR = 2e-4
CROP_SIZE = 21

# Reward Shaping
COLLISION_PENALTY = -1
SPEED_REWARD_THRESHOLD = 4
SPEED_REWARD = 0
BORDER_PENALTY_THRESHOLD = 5
BORDER_PENALTY = -0.01
PROGRESS_REWARD_SCALE = 1

# Bilan
SUMMARY_STEP_SIZE = 50

class SailingNet(nn.Module):
    """Architecture MLP pure : traite le voisinage local comme un capteur de proximité."""
    def __init__(self, crop_size=7, n_actions=9):
        super().__init__()
        self.crop_size = crop_size
        
        # Nombre de neurones d'entrée pour la carte locale (Terre, Wind_U, Wind_V)
        flattened_map_size = 3 * crop_size * crop_size
        
        # Branche de perception locale
        self.map_encoder = nn.Sequential(
            nn.Linear(flattened_map_size, 64),
            nn.ReLU()
        )
        
        # Branche scalaire (Vitesse, But, Delta Vent, Position Absolue)
        # On passe à 7 entrées pour aider l'agent avec les bordures (x_norm, y_norm)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU()
        )
        
        # Tête commune
        self.shared = nn.Sequential(
            nn.Linear(64 + 32, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(128, n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, local_map, scalars):
        # On aplatit le crop (batch, 3, 7, 7) -> (batch, 147)
        x_map = torch.flatten(local_map, start_dim=1)
        x_map = self.map_encoder(x_map)
        
        x_scalar = self.scalar_encoder(scalars)
        
        x = torch.cat([x_map, x_scalar], dim=1)
        x = self.shared(x)
        
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return probs, value

class Memory:
    """Buffer pour stocker les trajectoires avant l'update PPO."""
    def __init__(self):
        self.actions, self.states_map, self.states_scalars = [], [], []
        self.logprobs, self.rewards, self.is_terminals = [], [], []

    def clear(self):
        self.actions.clear(); self.states_map.clear(); self.states_scalars.clear()
        self.logprobs.clear(); self.rewards.clear(); self.is_terminals.clear()

class PPOSailingAgent:
    def __init__(self, crop_size=CROP_SIZE):
        self.crop_size = crop_size
        self.policy = SailingNet(crop_size=crop_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        # Décroissance du LR sur la durée totale
        self.scheduler = optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=NUM_EPISODES)
        
        self.policy_old = SailingNet(crop_size=crop_size)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.loss_history = {'actor': [], 'critic': []}

    def get_local_crop(self, obs):
        pos = obs[0:2].astype(int)
        wmap = obs[32774:49158].reshape(128, 128)
        wfield = obs[6:32774].reshape(128, 128, 2)
        pad = self.crop_size // 2
        
        # Padding constant pour les bords de map
        wmap_p = np.pad(wmap, pad, constant_values=1)
        wf_p = np.pad(wfield, ((pad, pad), (pad, pad), (0, 0)), constant_values=0)
        
        y, x = pos[1] + pad, pos[0] + pad
        crop = np.zeros((3, self.crop_size, self.crop_size))
        
        crop[0] = wmap_p[y-pad : y+pad+1, x-pad : x+pad+1]
        crop[1:] = wf_p[y-pad : y+pad+1, x-pad : x+pad+1].transpose(2, 0, 1)
        
        return torch.FloatTensor(crop).unsqueeze(0)

    def select_action(self, obs, goal, prev_angle):
        """Calcule les inputs et sélectionne une action selon la vieille politique."""
        with torch.no_grad():
            m_input = self.get_local_crop(obs)
            curr_w = obs[4:6]
            curr_a = np.arctan2(curr_w[1], curr_w[0])
            da = curr_a - prev_angle if prev_angle else 0

            x_norm = obs[0] / 128.0
            y_norm = obs[1] / 128.0
            
            s_input = torch.FloatTensor([obs[2], obs[3], goal[0]-obs[0], goal[1]-obs[1], da, x_norm, y_norm]).unsqueeze(0)
            
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
        
        returns = torch.tensor(rewards, dtype=torch.float32)

        # Conversion buffer
        map_s = torch.cat(memory.states_map); sca_s = torch.cat(memory.states_scalars)
        act_s = torch.tensor(memory.actions); lp_s = torch.stack(memory.logprobs).detach()

        for _ in range(5): # K-epochs
            probs, vals = self.policy(map_s, sca_s)
            dist = Categorical(probs)
            
            ratios = torch.exp(dist.log_prob(act_s) - lp_s)
            advantages = returns - vals.detach().squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
            
            a_loss = -torch.min(surr1, surr2).mean()
            c_loss = 0.5 * F.mse_loss(vals.squeeze(), returns)
            
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
history = {s: {'rewards': [], 'collision': [], 'success': [], 'steps': [], 'shaped_rewards': []} for s in SCENARIOS}
scenario_stats = {s: {'rewards': [], 'collision': [], 'success': [], 'steps': [], 'shaped_rewards': []} for s in SCENARIOS}
total_step = 0

for ep in tqdm(range(NUM_EPISODES)):
    scenario = SCENARIOS[ep % 3]
    env = SailingEnv(**get_wind_scenario(scenario))
    
    # --- Options de Reset ---
    options = {}
    if ep > RANDOM_START_EP:
        options["random_start"] = True
        options["wind_start_steps"] = int(np.random.randint(0, 100))

    if ep > RANDOM_VELO_EP:
        options["random_velocity"] = True
        
    obs, info = env.reset(options=options)
    goal = env.goal_position
    
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
        
        # --- Reward Shaping ---
        speed = np.linalg.norm(next_obs[2:4])
        curr_dist = np.linalg.norm(info['position'] - goal)
        x, y = info['position']
        
        # 1. Progression
        shaped_r = r + (p_dist - curr_dist) * PROGRESS_REWARD_SCALE
        
        # 2. Malus Stagnation
        if speed > SPEED_REWARD_THRESHOLD: 
            shaped_r += SPEED_REWARD
        
        # 3. Malus Bordure
        if x < BORDER_PENALTY_THRESHOLD or x > 128 - BORDER_PENALTY_THRESHOLD \
            or y < BORDER_PENALTY_THRESHOLD or y > 128 - BORDER_PENALTY_THRESHOLD:
            shaped_r += BORDER_PENALTY
            
        # 4. Collision
        collision = info.get('is_stuck', False)
        if collision:
            shaped_r = COLLISION_PENALTY
            done = True

        memory.states_map.append(m_in); memory.states_scalars.append(s_in)
        memory.actions.append(act); memory.logprobs.append(lp)
        memory.rewards.append(shaped_r); memory.is_terminals.append(done or trunc)

        obs, ep_r, p_dist = next_obs, ep_r + r, curr_dist

        if total_step % UPDATE_TIMESTEP == 0:
            agent.update(memory); memory.clear()
            
        if done or trunc: 
            break
    
    # --- Enregistrement des données ---
    agent.scheduler.step()
    
    history[scenario]['rewards'].append(ep_r)
    history[scenario]['shaped_rewards'].append(shaped_r)
    history[scenario]['collision'].append(1 if collision else 0)
    history[scenario]['success'].append(1 if (done and not collision) else 0)
    history[scenario]['steps'].append(ep_steps if not collision else None)
    
    scenario_stats[scenario]['rewards'].append(ep_r)
    scenario_stats[scenario]['shaped_rewards'].append(shaped_r)
    scenario_stats[scenario]['collision'].append(1 if collision else 0)
    scenario_stats[scenario]['success'].append(1 if (done and not collision) else 0)
    scenario_stats[scenario]['steps'].append(ep_steps if not collision else None)

    # --- Affichage Bilan ---
    if (ep + 1) % SUMMARY_STEP_SIZE == 0:
        curr_lr = agent.optimizer.param_groups[0]['lr']
        print(f"\n--- Bilan Eps {ep-28}-{ep+1} | LR: {curr_lr:.2e} ---")
        
        for s in SCENARIOS:
            s_data = scenario_stats[s]
            sr = np.mean(s_data['success']) * 100
            rw = np.mean(s_data['rewards'])
            srw = np.mean(s_data['shaped_rewards'])
            col = np.mean(s_data['collision']) * 100
            
            # Calcul de la moyenne des steps uniquement sur les valeurs non-None
            valid_steps = [st for st in s_data['steps'] if st is not None]
            
            if len(valid_steps) > 0:
                avg_step = np.mean(valid_steps)
                avg_collision = col
                step_str = f"{avg_step:4.1f}"
                indicator = "⭐" if avg_step < 60 else "📈" if avg_step < 90 else "🐌"
            else:
                step_str = "--- "
                indicator = "❌"
            
            print(f"[{s:10s}] Success: {sr:3.0f}% | Collision: {avg_collision:4.1f}% | Reward: {rw:5.1f} | Shaped Reward: {srw:5.1f} | Steps: {step_str} {indicator}")
            
        scenario_stats = {s: {'rewards': [], 'collision': [], 'success': [], 'steps': [], 'shaped_rewards': []} for s in SCENARIOS}

agent.save("mlp_model.pth")