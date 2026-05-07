"""
AWAC-lite Sailing Agent
=======================

Idée principale
----------------
On ne veut PAS réapprendre une politique from scratch.
On veut :

    windmaster + améliorations locales RL

Donc :
    1. gros dataset expert
    2. behavioral cloning fort
    3. fine-tuning AWAC-lite

Différence majeure vs PPO :
- pas de clipping PPO
- pas d'entropy bonus
- pas d'exploration agressive
- pas de policy collapse

L'acteur reste proche de l'expert.
"""

from collections import deque
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.env_sailing import SailingEnv
from src.wind_scenarios import get_wind_scenario
from src.agents.windmaster import MyAgent


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCENARIOS = [
    "training_1",
    "training_2",
    "training_3",
]

DEFAULT_START = [64, 0]

# Dataset
EXPERT_EPISODES = 500

# Training
NUM_EPISODES = 3000
BATCH_SIZE = 256
BUFFER_SIZE = 300_000

# RL
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

# AWAC
LAMBDA_AWAC = 0.3
BC_PRETRAIN_EPOCHS = 40

# Exploration
EXPERT_MIX_START = 0.80
EXPERT_MIX_END = 0.10

# Reward
GOAL_REWARD = 20.0
COLLISION_PENALTY = -20.0
STEP_PENALTY = -0.01
DIST_REWARD = 0.1

# Features
CROP_SIZE = 17

# Updates
UPDATES_PER_EPISODE = 20


# ============================================================
# NETWORK
# ============================================================

class SailingNet(nn.Module):

    def __init__(self, crop_size=17, n_scalars=9, n_actions=9):
        super().__init__()

        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),

            nn.Linear(288, 128),
            nn.ReLU(),
        )

        self.scalar_encoder = nn.Sequential(
            nn.Linear(n_scalars, 64),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(128 + 64, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.actor = nn.Linear(128, n_actions)
        self.q_head = nn.Linear(128, n_actions)

    def forward(self, map_obs, scalar_obs):

        m = self.map_encoder(map_obs)
        s = self.scalar_encoder(scalar_obs)

        x = torch.cat([m, s], dim=1)
        x = self.shared(x)

        logits = self.actor(x)
        qvals = self.q_head(x)

        return logits, qvals


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)

        maps = torch.cat([b[0] for b in batch]).to(DEVICE)
        scalars = torch.cat([b[1] for b in batch]).to(DEVICE)
        actions = torch.tensor([b[2] for b in batch]).long().to(DEVICE)
        rewards = torch.tensor([b[3] for b in batch]).float().to(DEVICE)
        next_maps = torch.cat([b[4] for b in batch]).to(DEVICE)
        next_scalars = torch.cat([b[5] for b in batch]).to(DEVICE)
        dones = torch.tensor([b[6] for b in batch]).float().to(DEVICE)

        return (
            maps,
            scalars,
            actions,
            rewards,
            next_maps,
            next_scalars,
            dones,
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# AGENT
# ============================================================

class AWACSailingAgent:

    def __init__(self):

        self.net = SailingNet(CROP_SIZE).to(DEVICE)
        self.target_net = SailingNet(CROP_SIZE).to(DEVICE)

        self.target_net.load_state_dict(self.net.state_dict())

        self.actor_optimizer = optim.Adam(
            self.net.actor.parameters(),
            lr=LR_ACTOR
        )

        self.critic_optimizer = optim.Adam(
            list(self.net.map_encoder.parameters()) +
            list(self.net.scalar_encoder.parameters()) +
            list(self.net.shared.parameters()) +
            list(self.net.q_head.parameters()),
            lr=LR_CRITIC
        )

    # ========================================================
    # FEATURES
    # ========================================================

    def get_local_crop(self, obs):

        pos = obs[0:2].astype(int)

        wmap = obs[32774:49158].reshape(128, 128)
        wfield = obs[6:32774].reshape(128, 128, 2)

        pad = CROP_SIZE // 2

        wmap_p = np.pad(wmap, pad, constant_values=1)
        wf_p = np.pad(
            wfield,
            ((pad, pad), (pad, pad), (0, 0)),
            constant_values=0
        )

        y, x = pos[1] + pad, pos[0] + pad

        crop = np.zeros((3, CROP_SIZE, CROP_SIZE), dtype=np.float32)

        crop[0] = wmap_p[y-pad:y+pad+1, x-pad:x+pad+1]

        crop[1:] = wf_p[
            y-pad:y+pad+1,
            x-pad:x+pad+1
        ].transpose(2, 0, 1)

        return torch.FloatTensor(crop).unsqueeze(0)

    def build_scalars(self, obs, goal):

        dx = goal[0] - obs[0]
        dy = goal[1] - obs[1]

        dist = np.sqrt(dx**2 + dy**2)

        angle = np.arctan2(dy, dx)

        s = torch.FloatTensor([
            obs[2] / 10.0,
            obs[3] / 10.0,

            dx / 128.0,
            dy / 128.0,

            obs[4] / 10.0,
            obs[5] / 10.0,

            np.sin(angle),
            np.cos(angle),

            dist / 181.0,
        ]).unsqueeze(0)

        return s

    # ========================================================
    # ACTION
    # ========================================================

    @torch.no_grad()
    def act(self, map_obs, scalar_obs):

        logits, _ = self.net(map_obs, scalar_obs)

        probs = torch.softmax(logits, dim=1)

        action = torch.multinomial(probs, 1)

        return action.item()

    # ========================================================
    # BC
    # ========================================================

    def behavioral_cloning(self, dataset):

        print("\n=== Behavioral Cloning ===")

        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        maps = torch.cat([d[0] for d in dataset]).to(DEVICE)
        scalars = torch.cat([d[1] for d in dataset]).to(DEVICE)
        actions = torch.tensor([d[2] for d in dataset]).long().to(DEVICE)

        N = len(actions)

        for epoch in range(BC_PRETRAIN_EPOCHS):

            idx = np.random.permutation(N)

            total_loss = 0

            for s in range(0, N, BATCH_SIZE):

                mb = idx[s:s+BATCH_SIZE]

                logits, _ = self.net(
                    maps[mb],
                    scalars[mb]
                )

                loss = F.cross_entropy(logits, actions[mb])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"BC Epoch {epoch+1:02d} "
                f"| Loss {total_loss:.4f}"
            )

        self.target_net.load_state_dict(
            self.net.state_dict()
        )

    # ========================================================
    # AWAC UPDATE
    # ========================================================

    def update(self, replay):

        if len(replay) < BATCH_SIZE:
            return

        for _ in range(UPDATES_PER_EPISODE):

            (
                maps,
                scalars,
                actions,
                rewards,
                next_maps,
                next_scalars,
                dones,
            ) = replay.sample(BATCH_SIZE)

            # =================================================
            # TARGET Q
            # =================================================

            with torch.no_grad():

                next_logits, next_q = self.target_net(
                    next_maps,
                    next_scalars
                )

                next_probs = torch.softmax(next_logits, dim=1)

                next_v = (
                    next_probs * next_q
                ).sum(dim=1)

                target_q = (
                    rewards +
                    GAMMA * (1 - dones) * next_v
                )

            # =================================================
            # CRITIC
            # =================================================

            _, qvals = self.net(
                maps,
                scalars
            )

            q_a = qvals.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze(1)

            critic_loss = F.mse_loss(
                q_a,
                target_q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # =================================================
            # ADVANTAGE
            # =================================================

            with torch.no_grad():

                logits, qvals = self.net(
                    maps,
                    scalars
                )

                probs = torch.softmax(logits, dim=1)

                v = (
                    probs * qvals
                ).sum(dim=1)

                adv = q_a - v

                weights = torch.exp(
                    adv / LAMBDA_AWAC
                ).clamp(max=20)

            # =================================================
            # ACTOR
            # =================================================

            logits, _ = self.net(
                maps,
                scalars
            )

            log_probs = F.log_softmax(
                logits,
                dim=1
            )

            chosen_log_probs = log_probs.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze(1)

            actor_loss = -(
                weights * chosen_log_probs
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # =================================================
            # TARGET UPDATE
            # =================================================

            for tp, p in zip(
                self.target_net.parameters(),
                self.net.parameters()
            ):
                tp.data.copy_(
                    TAU * p.data +
                    (1 - TAU) * tp.data
                )

    # ========================================================
    # SAVE
    # ========================================================

    def save(self, path):

        torch.save(
            self.net.state_dict(),
            path
        )


# ============================================================
# DATASET EXPERT
# ============================================================

def compute_reward(
    prev_dist,
    curr_dist,
    done,
    collision
):

    reward = STEP_PENALTY

    reward += (
        prev_dist - curr_dist
    ) * DIST_REWARD

    if collision:
        reward += COLLISION_PENALTY

    if done and not collision:
        reward += GOAL_REWARD

    return reward


def collect_expert_dataset(agent):

    expert = MyAgent()

    dataset = []

    print("\n=== Collecting expert dataset ===")

    for ep in tqdm(range(EXPERT_EPISODES)):

        scenario = SCENARIOS[ep % 3]

        env = SailingEnv(
            **get_wind_scenario(scenario)
        )

        obs, _ = env.reset(
            options={
                "start_position": DEFAULT_START
            }
        )

        goal = env.goal_position

        prev_dist = np.linalg.norm(
            obs[0:2] - goal
        )

        for t in range(500):

            map_obs = agent.get_local_crop(obs)
            scalar_obs = agent.build_scalars(obs, goal)

            action = expert.act(obs)

            next_obs, _, done, trunc, info = env.step(action)

            curr_dist = np.linalg.norm(
                next_obs[0:2] - goal
            )

            collision = info.get(
                "is_stuck",
                False
            )

            reward = compute_reward(
                prev_dist,
                curr_dist,
                done,
                collision
            )

            next_map = agent.get_local_crop(next_obs)

            next_scalar = agent.build_scalars(
                next_obs,
                goal
            )

            dataset.append((
                map_obs,
                scalar_obs,
                action,
                reward,
                next_map,
                next_scalar,
                done or trunc
            ))

            obs = next_obs
            prev_dist = curr_dist

            if done or trunc:
                break

    return dataset


# ============================================================
# MAIN
# ============================================================

agent = AWACSailingAgent()

# ============================================================
# DATASET EXPERT
# ============================================================

expert_dataset = collect_expert_dataset(agent)

# ============================================================
# BC
# ============================================================

agent.behavioral_cloning(expert_dataset)

# ============================================================
# REPLAY
# ============================================================

replay = ReplayBuffer(BUFFER_SIZE)

for transition in expert_dataset:
    replay.add(transition)

# ============================================================
# RL
# ============================================================

expert = MyAgent()

print("\n=== AWAC TRAINING ===")

for ep in tqdm(range(NUM_EPISODES)):

    scenario = SCENARIOS[
        ep % len(SCENARIOS)
    ]

    env = SailingEnv(
        **get_wind_scenario(scenario)
    )

    obs, _ = env.reset(
        options={
            "start_position": DEFAULT_START
        }
    )

    goal = env.goal_position

    prev_dist = np.linalg.norm(
        obs[0:2] - goal
    )

    expert_ratio = (
        EXPERT_MIX_START +
        (EXPERT_MIX_END - EXPERT_MIX_START)
        * (ep / NUM_EPISODES)
    )

    success = False
    collision = False

    for t in range(500):

        map_obs = agent.get_local_crop(obs)
        scalar_obs = agent.build_scalars(
            obs,
            goal
        )

        # ====================================================
        # EXPERT MIXING
        # ====================================================

        if random.random() < expert_ratio:
            action = expert.act(obs)
        else:
            action = agent.act(
                map_obs.to(DEVICE),
                scalar_obs.to(DEVICE)
            )

        next_obs, _, done, trunc, info = env.step(action)

        curr_dist = np.linalg.norm(
            next_obs[0:2] - goal
        )

        collision = info.get(
            "is_stuck",
            False
        )

        reward = compute_reward(
            prev_dist,
            curr_dist,
            done,
            collision
        )

        next_map = agent.get_local_crop(next_obs)
        next_scalar = agent.build_scalars(
            next_obs,
            goal
        )

        replay.add((
            map_obs,
            scalar_obs,
            action,
            reward,
            next_map,
            next_scalar,
            done or trunc
        ))

        obs = next_obs
        prev_dist = curr_dist

        if done and not collision:
            success = True

        if done or trunc:
            break

    agent.update(replay)

    if (ep + 1) % 50 == 0:

        print(
            f"\nEpisode {ep+1}"
            f" | ExpertMix {expert_ratio:.2f}"
            f" | Success {success}"
            f" | Collision {collision}"
            f" | Replay {len(replay)}"
        )

# ============================================================
# SAVE
# ============================================================

agent.save("awac_sailing.pth")

print("\nTraining finished.")