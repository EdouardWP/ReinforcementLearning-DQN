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

VERSION_NAME = 'DQN_v03'

# Constants for training - smaller episodes for faster iterations
REPORT_EPISODES = 100  # Reduced for more frequent updates
DISPLAY_EPISODES = 50   # Reduced for performance
NUM_EPISODES = 20_000   # Reduced total episodes
MAX_T = 2000

# DDPG specific parameters - optimized for faster learning
BATCH_SIZE = 64     # Smaller batch for faster updates
MEMORY_SIZE = 20000  # Larger memory for better experience diversity
GAMMA = 0.99       # Discount factor
TAU = 0.01         # Faster target network update rate
ACTOR_LR = 3e-4    # Increased learning rate
CRITIC_LR = 3e-3   # Increased learning rate
NOISE_SIGMA = 0.1  # Reduced exploration noise for more stable actions
UPDATE_EVERY = 2   # Update networks every N steps

# Define transition tuple for memory management
Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward', 'done'))

# Optimized Ornstein-Uhlenbeck noise process
class OUNoise:
    """Ornstein-Uhlenbeck process for generating correlated noise"""
    def __init__(self, action_dimension, mu=0, theta=0.2, sigma=0.1):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu
        
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# Actor network - simplified and optimized
class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):  # Reduced hidden size
        super(Actor, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        # Better weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x):
        # Ensure input has correct dimensions
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if missing
            
        # Check input dimensions
        if x.size(1) != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {x.size(1)}")
            
        # Apply layers with leaky ReLU for better gradient flow
        x = F.leaky_relu(self.layer1(x), 0.1)
        x = F.leaky_relu(self.layer2(x), 0.1)
        
        # Output is tanh to bound actions between -1 and 1
        return torch.tanh(self.layer3(x))

# Critic network - simplified and optimized
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):  # Reduced hidden size
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Process state
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # Combine state and action
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        
        # Output Q-value
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Better weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, state, action):
        # Ensure input has correct dimensions
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Check input dimensions
        if state.size(1) != self.state_size:
            raise ValueError(f"Expected state size {self.state_size}, got {state.size(1)}")
        if action.size(1) != self.action_size:
            raise ValueError(f"Expected action size {self.action_size}, got {action.size(1)}")
            
        # Process state
        xs = F.leaky_relu(self.fc1(state), 0.1)
        
        # Combine state and action
        x = torch.cat([xs, action], dim=1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        
        # Output Q-value
        return self.fc3(x)

# Replay Buffer with prioritized sampling
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.alpha = 0.6  # Priority exponent

    def push(self, state, action, next_state, reward, done):
        """Save a transition with maximum priority"""
        max_priority = max(self.priorities, default=1.0)
        self.memory.append(Transition(state, action, next_state, reward, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return random.sample(self.memory, len(self.memory))
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        return samples

    def __len__(self):
        return len(self.memory)

# DDPG Agent with optimizations
class DDPGAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.t_step = 0
        
        # Actor networks
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        
        # Critic networks
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=CRITIC_LR, weight_decay=1e-4)  # Added weight decay
        
        # Noise process for exploration with adaptive noise
        self.noise = OUNoise(action_size, sigma=NOISE_SIGMA)
        self.noise_scale = 1.0  # Will be annealed during training
        
        # Memory
        self.memory = ReplayBuffer(MEMORY_SIZE)
        
        # Training stats
        self.actor_loss_history = []
        self.critic_loss_history = []
    
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure state has correct dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Verify state shape
        if state.shape[1] != self.state_size:
            # Try to reshape or pad if needed
            if len(state.flatten()) >= self.state_size:
                state = state.flatten()[:self.state_size].unsqueeze(0)
            else:
                # Pad with zeros
                padding = torch.zeros(1, self.state_size - state.shape[1], device=self.device)
                state = torch.cat([state, padding], dim=1)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample() * noise_scale
            action += noise
        
        # Clip action to be within valid range
        return np.clip(action, -1.0, 1.0)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.push(state, action, next_state, reward, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self.learn(experiences)
    
    def learn(self, experiences):
        batch = Transition(*zip(*experiences))
        
        # Convert to tensors
        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array(batch.reward), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array(batch.done), dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
            
        # Get expected Q values
        Q_expected = self.critic(states, actions)
        
        # Compute critic loss - Huber loss for stability
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # Gradient clipping
        self.critic_optimizer.step()
        
        # Update actor
        actions_pred = self.actor(states)
        actor_loss = -self.critic(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # Gradient clipping
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update_targets()
        
        # Record loss
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
    
    def soft_update_targets(self):
        """Soft update target networks"""
        # Update target networks using soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

def simulate(learning=True, episode_start=0):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create models directory
    if not os.path.exists(f'models_{VERSION_NAME}'): 
        os.makedirs(f'models_{VERSION_NAME}')
    
    # Create environment with continuous state and action space
    env = gym.make("Pyrace-v3").unwrapped
    
    # Get dimensions for network
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # Initialize agent
    agent = DDPGAgent(state_size, action_size, device)
    
    # Load previous model if continuing training
    if episode_start > 0 and learning:
        try:
            print(f"Loading model from episode {episode_start}")
            agent.load(f'models_{VERSION_NAME}/ddpg_model_{episode_start}.pth')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Could not load model: {e}. Starting fresh.")
    
    # Tracking variables
    total_rewards = []
    max_reward = -10_000
    episode_duration = []
    
    # Enable rendering only for display episodes to improve performance
    env.set_view(False)
    
    # Define noise annealing schedule
    noise_start = 1.0
    noise_end = 0.1
    noise_decay = NUM_EPISODES / 2  # Decay over half the episodes
    
    # Training loop
    for episode in range(episode_start, NUM_EPISODES + episode_start):
        # Reset environment
        state, _ = env.reset()
        agent.noise.reset()
        
        # Set noise scale based on episode
        noise_scale = noise_end + (noise_start - noise_end) * math.exp(-1. * episode / noise_decay)
        
        # Enable rendering only for display episodes or play mode
        if episode % DISPLAY_EPISODES == 0 or not learning:
            env.set_view(True)
            if not learning:
                env.pyrace.mode = 2  # continuous display for play mode
        else:
            env.set_view(False)  # Disable rendering for most episodes
            
        episode_reward = 0
        
        # Episode loop
        for t in range(MAX_T):
            # Select and perform action
            action = agent.select_action(state, add_noise=learning, noise_scale=noise_scale)
            next_state, reward, done, _, info = env.step(action)
            
            if learning:
                # Store transition and learn
                agent.step(state, action, reward, next_state, done)
            
            # Update state and accumulate reward
            state = next_state
            episode_reward += reward
            
            # Display progress only when rendering
            if env.is_view:
                env.set_msgs(['SIMULATE',
                            f'Episode: {episode}',
                            f'Time steps: {t}',
                            f'Speed: {info["speed"]:.1f}',
                            f'Check: {info["check"]}',
                            f'Crash: {info["crash"]}',
                            f'Reward: {episode_reward:.1f}',
                            f'Max Reward: {max_reward:.1f}',
                            f'Noise: {noise_scale:.2f}'])
                env.render()
            
            # Check if episode is done
            if done or t >= MAX_T - 1:
                episode_duration.append(t)
                total_rewards.append(episode_reward)
                
                if episode_reward > max_reward: 
                    max_reward = episode_reward
                    if learning:
                        agent.save(f'models_{VERSION_NAME}/best_model.pth')
                
                print(f"Episode {episode} finished after {t} steps with reward {episode_reward:.1f}")
                break
        
        # Save more frequently at the beginning, then less often
        save_interval = REPORT_EPISODES if episode > 1000 else 100
        if learning and episode > 0 and episode % save_interval == 0:
            # Save model
            agent.save(f'models_{VERSION_NAME}/ddpg_model_{episode}.pth')
            print(f"Model saved at episode {episode}")
            
            # Plot progress every few save points to reduce overhead
            if episode % (save_interval * 5) == 0:
                plt.figure(figsize=(15, 10))
                
                # Plot rewards
                plt.subplot(2, 2, 1)
                plt.plot(total_rewards[-500:])  # Show only recent rewards for clarity
                plt.ylabel('rewards')
                plt.xlabel('episode')
                plt.title(f'Recent Training rewards - episode {episode}')
                
                # Plot episode duration
                plt.subplot(2, 2, 2)
                plt.plot(episode_duration[-500:])  # Show only recent durations
                plt.ylabel('steps')
                plt.xlabel('episode')
                plt.title('Recent Episode durations')
                
                if len(agent.actor_loss_history) > 0:
                    # Plot actor loss
                    plt.subplot(2, 2, 3)
                    plt.plot(agent.actor_loss_history[-1000:])
                    plt.ylabel('actor loss')
                    plt.xlabel('optimization step')
                    plt.title('Actor Loss (last 1000 steps)')
                    
                    # Plot critic loss
                    plt.subplot(2, 2, 4)
                    plt.plot(agent.critic_loss_history[-1000:])
                    plt.ylabel('critic loss')
                    plt.xlabel('optimization step')
                    plt.title('Critic Loss (last 1000 steps)')
                
                plt.tight_layout()
                plt.savefig(f'models_{VERSION_NAME}/training_progress_{episode}.png')
                plt.close()
    
    # Final save
    if learning:
        agent.save(f'models_{VERSION_NAME}/final_model.pth')
    
    # Close environment
    env.close()
    return max_reward


def load_and_play(episode=None):
    """Load a trained model and play the game"""
    # If episode not specified, load best model
    model_path = f'models_{VERSION_NAME}/best_model.pth' if episode is None else f'models_{VERSION_NAME}/ddpg_model_{episode}.pth'
    
    # Create environment
    env = gym.make("Pyrace-v3").unwrapped
    env.set_view(True)
    env.pyrace.mode = 2  # continuous display of game
    
    # Get dimensions for network
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    # Initialize agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(state_size, action_size, device)
    
    # Load model
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Play one episode
    state, _ = env.reset()
    done = False
    total_reward = 0
    t = 0
    
    while not done and t < MAX_T:
        # Select action without noise
        action = agent.select_action(state, add_noise=False)
        
        # Take action
        next_state, reward, done, _, info = env.step(action)
        
        # Update state and accumulate reward
        state = next_state
        total_reward += reward
        t += 1
        
        # Display
        env.set_msgs(['PLAY MODE',
                    f'Time steps: {t}',
                    f'Speed: {info["speed"]:.1f}',
                    f'Check: {info["check"]}',
                    f'Crash: {info["crash"]}',
                    f'Reward: {total_reward:.1f}',
                    f'Action: [{action[0]:.2f}, {action[1]:.2f}]'])
        env.render()
    
    print(f"Episode finished with reward {total_reward} after {t} steps")
    return total_reward


if __name__ == "__main__":
    if len(sys.argv) == 2:
        if sys.argv[1] == "train":
            simulate(learning=True)
        elif sys.argv[1] == "play":
            load_and_play()
        elif sys.argv[1].startswith("play_"):
            # Format: play_10000 to play from a specific episode
            episode = int(sys.argv[1].split("_")[1])
            load_and_play(episode)
        elif sys.argv[1].startswith("train_"):
            # Format: train_10000 to continue training from episode 10000
            episode_start = int(sys.argv[1].split("_")[1])
            simulate(learning=True, episode_start=episode_start)
    else:
        # Default: train
        simulate(learning=True) 