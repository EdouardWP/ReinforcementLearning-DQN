import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # to avoid some "memory" errors with TkAgg backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

import gymnasium as gym
import gym_race

VERSION_NAME = 'DQN_v01'

# Constants matching Q-table version
REPORT_EPISODES = 500
DISPLAY_EPISODES = 100
NUM_EPISODES = 65_000
MAX_T = 2000

# Improved DQN specific parameters
BATCH_SIZE = 128
MEMORY_SIZE = 10000
GAMMA = 0.99  # Discount factor
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005  # Target network update rate
LEARNING_RATE = 1e-4

# Define transition tuple for better memory management
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Improved neural network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_size)
        
        # Better weight initialization
        nn.init.kaiming_normal_(self.layer1.weight)
        nn.init.kaiming_normal_(self.layer2.weight)
        nn.init.kaiming_normal_(self.layer3.weight)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Improved ReplayBuffer using namedtuple
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

steps_done = 0

def select_action(state, policy_net, device, n_actions, evaluation=False):
    global steps_done
    # Convert state to tensor if it's not already
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
    # Epsilon greedy action selection
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold or evaluation:
        with torch.no_grad():
            # Choose action with highest expected value
            return policy_net(state).max(1)[1].view(1, 1).item()
    else:
        return random.randrange(n_actions)

def optimize_model(policy_net, target_net, optimizer, memory, device):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Create mask for non-final states
    non_final_mask = torch.tensor([not done for done in batch.done], 
                                 device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0) 
                                     for s, done in zip(batch.next_state, batch.done) if not done])
    
    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=device)
    action_batch = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=device)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss (smooth L1)
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    
    return loss.item()

def simulate(learning=True, episode_start=0):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make("Pyrace-v1").unwrapped
    if not os.path.exists(f'models_{VERSION_NAME}'): 
        os.makedirs(f'models_{VERSION_NAME}')
    
    # Get dimensions for network
    n_actions = env.action_space.n
    n_observations = len(env.observation_space.low)
    
    # Initialize DQN and target network
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # Set target network to evaluation mode
    
    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayBuffer(MEMORY_SIZE)
    
    # Load previous model if continuing training
    if episode_start > 0 and learning:
        try:
            print(f"Loading model from episode {episode_start}")
            checkpoint = torch.load(f'models_{VERSION_NAME}/dqn_model_{episode_start}.pth')
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            target_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            global steps_done
            steps_done = checkpoint.get('steps_done', 0)
        except:
            print(f"Could not load model from episode {episode_start}, starting fresh")
    
    total_rewards = []
    max_reward = -10_000
    
    env.set_view(True)
    
    for episode in range(episode_start, NUM_EPISODES + episode_start):
        # Reset environment
        obv, _ = env.reset()
        total_reward = 0
        last_check = 0
        
        if not learning:
            env.pyrace.mode = 2  # continuous display of game
        
        if episode > 0:
            total_rewards.append(total_reward)
            
            if learning and episode % REPORT_EPISODES == 0:
                plt.figure(figsize=(10, 5))
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.xlabel('episode')
                plt.title(f'Training progress - episode {episode}')
                plt.show(block=False)
                plt.pause(4.0)
                
                # Save model
                torch.save({
                    'model_state_dict': policy_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'episode': episode,
                    'steps_done': steps_done
                }, f'models_{VERSION_NAME}/dqn_model_{episode}.pth')
                plt.close()
        
        for t in range(MAX_T):
            # Select and perform action
            action = select_action(obv, policy_net, device, n_actions, evaluation=not learning)
            next_obv, reward, done, _, info = env.step(action)
            
            # Reward shaping
            if info["crash"]:
                reward = -10  # Penalty for crashing
            elif info["check"] > last_check:
                # Big reward for reaching new checkpoint
                reward += 25 * (info["check"] - last_check)
                last_check = info["check"]
            
            # Store transition in memory
            if learning:
                memory.push(obv, action, next_obv, reward, done)
                
                # Perform optimization step
                loss = optimize_model(policy_net, target_net, optimizer, memory, device)
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)
            
            total_reward += reward
            obv = next_obv
            
            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
                
                env.set_msgs(['SIMULATE',
                            f'Episode: {episode}',
                            f'Time steps: {t}',
                            f'check: {info["check"]}',
                            f'dist: {info["dist"]}',
                            f'crash: {info["crash"]}',
                            f'Reward: {total_reward:.0f}',
                            f'Max Reward: {max_reward:.0f}',
                            f'Epsilon: {eps_threshold:.2f}'])
                env.render()
            
            if done or t >= MAX_T - 1:
                if total_reward > max_reward: 
                    max_reward = total_reward
                break
    
    print("Training complete!")

def load_and_play(episode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Pyrace-v1").unwrapped
    
    # Get dimensions for network
    n_actions = env.action_space.n
    n_observations = len(env.observation_space.low)
    
    policy_net = DQN(n_observations, n_actions).to(device)
    
    # Load model
    checkpoint = torch.load(f'models_{VERSION_NAME}/dqn_model_{episode}.pth')
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.eval()  # Set to evaluation mode
    
    simulate(learning=False, episode_start=episode)

if __name__ == "__main__":
    # Start training from scratch with new architecture
    simulate(learning=True, episode_start=3500)  
    # load_and_play(500)  # Uncomment this after you have a trained model 