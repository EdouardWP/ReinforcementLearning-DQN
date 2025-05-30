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

VERSION_NAME = 'DQN_v03_opt'

# Constants for training - further optimized for faster convergence
REPORT_EPISODES = 50  # More frequent updates
DISPLAY_EPISODES = 25  # Less frequent display
NUM_EPISODES = 5_000   # Far fewer episodes needed with optimizations
MAX_T = 2000

# DDPG specific parameters - hyper-optimized
BATCH_SIZE = 128       # Larger batch size for better gradient estimates
MEMORY_SIZE = 50000    # Much larger memory for better retention of rare experiences
GAMMA = 0.99           # Discount factor
TAU = 0.05             # Even faster target network update rate
ACTOR_LR = 5e-4        # Further increased learning rate
CRITIC_LR = 5e-3       # Further increased learning rate
NOISE_SIGMA = 0.15     # Balanced exploration
UPDATE_EVERY = 1       # Update on every step for faster learning
PER_ALPHA = 0.7        # Prioritized experience replay exponent
PER_BETA = 0.5         # Initial importance sampling weight
N_STEP_RETURNS = 3     # Use n-step returns for faster credit assignment

# Define transition tuple for memory management
Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

# Optimized Gaussian noise process (simpler than OU)
class GaussianNoise:
    """Simple Gaussian noise for exploration"""
    def __init__(self, action_dimension, sigma=0.15):
        self.action_dimension = action_dimension
        self.sigma = sigma
        
    def reset(self):
        pass
        
    def sample(self):
        return np.random.normal(0, self.sigma, self.action_dimension)

# Actor network - optimized architecture
class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):  # Larger network for better representation
        super(Actor, self).__init__()
        self.input_size = input_size
        
        # Normalize inputs
        self.bn_input = nn.BatchNorm1d(input_size)
        
        # Main network with more hidden units
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Better weight initialization - scaled for ELU activation
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Initialize with He initialization (good for ELU/ReLU)
            nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu')
            if module.bias is not None:
                module.bias.data.zero_()
        
    def forward(self, x):
        # Ensure input has correct dimensions
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
            
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # BatchNorm handles single samples differently during training vs eval
        training_mode = self.training and x.size(0) > 1
        
        # Use batch normalization conditionally
        if x.size(0) > 1:
            x = self.bn_input(x)
            
        # Apply layers with ELU for better gradient flow
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        
        # Output is tanh to bound actions between -1 and 1
        return torch.tanh(self.fc3(x))

# Critic network - optimized architecture with dual streams
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # State path
        self.bn_state = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, hidden_size)
        
        # Combined path after state processing
        self.fc2 = nn.Linear(hidden_size + action_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
        # Layer norm for hidden layers (more stable than batch norm for critic)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        # Better weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu')
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
            
        # Use batch normalization conditionally
        if state.size(0) > 1:
            state = self.bn_state(state)
            
        # Process state path
        xs = F.elu(self.fc1(state))
        xs = self.ln1(xs)  # Layer normalization
        
        # Combine state and action
        x = torch.cat([xs, action], dim=1)
        x = F.elu(self.fc2(x))
        
        # Output Q-value
        return self.fc3(x)

# Improved Replay Buffer with N-step returns and prioritized experience replay
class NStepPrioritizedReplayBuffer:
    def __init__(self, capacity, n_steps=1, gamma=0.99, alpha=0.6, beta=0.4):
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.n_steps = n_steps
        self.gamma = gamma
        self.alpha = alpha  # How much to prioritize
        self.beta = beta    # Importance sampling weight
        self.beta_increment = 0.001  # How much to increase beta over time
        self.n_step_buffer = deque(maxlen=n_steps)
        self.max_priority = 1.0
        
    def _get_n_step_info(self):
        """Returns the n-step reward, next_state, and done"""
        reward, next_state, done = self.n_step_buffer[-1][-3:]
        
        # Calculate n-step rewards
        for i in range(len(self.n_step_buffer) - 1, 0, -1):
            r, s, d = self.n_step_buffer[i-1][-3:]
            reward = r + self.gamma * reward * (1 - d)
            if d:
                next_state = s
                done = True
                
        return reward, next_state, done

    def add(self, state, action, reward, next_state, done):
        """Add experience to n-step buffer"""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return
            
        # Get the n-step returns
        if len(self.n_step_buffer) >= self.n_steps:
            state, action = self.n_step_buffer[0][:2]
            reward, next_state, done = self._get_n_step_info()
            
            # Add to memory with maximum priority
            self.memory.append((state, action, reward, next_state, done))
            self.priorities.append(self.max_priority)
            
    def sample(self, batch_size):
        """Sample a batch of experiences with priorities"""
        # Increase beta over time for annealing importance sampling bias
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate sampling probabilities
        if len(self.memory) == batch_size:
            indices = range(len(self.memory))
        else:
            priorities = np.array(self.priorities)
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            
            # Sample based on priorities
            indices = np.random.choice(len(self.memory), batch_size, p=probabilities, replace=False)
        
        # Calculate importance sampling weights
        weights = []
        p_min = min(self.priorities) / sum(self.priorities)
        max_weight = (p_min * len(self.memory)) ** (-self.beta)
        
        samples = []
        for i in indices:
            p_sample = self.priorities[i] / sum(self.priorities)
            weight = (p_sample * len(self.memory)) ** (-self.beta)
            weights.append(weight / max_weight)  # Normalize weights
            
            state, action, reward, next_state, done = self.memory[i]
            samples.append(Transition(state, action, reward, next_state, done, i))
            
        return samples, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD error"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.memory)

# DDPG Agent with further optimizations
class DDPGAgent:
    def __init__(self, state_size, action_size, device):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.t_step = 0
        
        # Use EMA (Exponential Moving Average) tracking for more stable learning
        self.ema_reward = None
        
        # Actor networks
        self.actor = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Use Adam with weight decay and custom parameters
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=ACTOR_LR,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # Critic networks
        self.critic = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), 
            lr=CRITIC_LR, 
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        
        # Better noise process for exploration
        self.noise = GaussianNoise(action_size, sigma=NOISE_SIGMA)
        
        # Memory with n-step returns and prioritization
        self.memory = NStepPrioritizedReplayBuffer(
            MEMORY_SIZE, 
            n_steps=N_STEP_RETURNS, 
            gamma=GAMMA,
            alpha=PER_ALPHA,
            beta=PER_BETA
        )
        
        # Training stats
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.td_error_history = []
        
        # Adaptive learning rate scheduling
        self.lr_scheduler_actor = optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
        self.lr_scheduler_critic = optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, mode='max', factor=0.5, patience=100, verbose=True
        )
    
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        
        # Ensure state has correct dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Check state shape
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
        
        # Add noise for exploration
        if add_noise:
            noise = self.noise.sample() * noise_scale
            action += noise
        
        # Clip action to be within valid range
        return np.clip(action, -1.0, 1.0)
    
    def step(self, state, action, reward, next_state, done):
        # Normalize rewards for more stable learning
        if self.ema_reward is None:
            self.ema_reward = reward
        else:
            self.ema_reward = 0.99 * self.ema_reward + 0.01 * reward
        
        # Add experience to replay buffer
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps if enough samples
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            self.learn()
    
    def learn(self):
        # Sample from replay buffer
        experiences, is_weights = self.memory.sample(BATCH_SIZE)
        
        # Convert to tensors
        states = torch.tensor(np.array([e.state for e in experiences]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([e.action for e in experiences]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([e.reward for e in experiences]), dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array([e.next_state for e in experiences]), dtype=torch.float32, device=self.device)
        dones = torch.tensor(np.array([e.done for e in experiences]), dtype=torch.float32, device=self.device).unsqueeze(1)
        indices = np.array([e.priority for e in experiences])
        is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = rewards + (GAMMA**N_STEP_RETURNS) * Q_targets_next * (1 - dones)
            
        # Get expected Q values
        Q_expected = self.critic(states, actions)
        
        # Calculate TD errors for updating priorities
        td_errors = torch.abs(Q_targets - Q_expected).detach().cpu().numpy()
        
        # Compute critic loss with importance sampling weights
        critic_loss = (is_weights * F.smooth_l1_loss(Q_expected, Q_targets, reduction='none')).mean()
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)  # Gradient clipping
        self.critic_optimizer.step()
        
        # Update actor (delayed policy updates)
        actions_pred = self.actor(states)
        actor_loss = -(is_weights * self.critic(states, actions_pred)).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)  # Gradient clipping
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update_targets()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors.flatten() + 1e-5)  # small constant for stability
        
        # Record loss and TD error
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
        self.td_error_history.append(np.mean(td_errors))
    
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
            'actor_lr_scheduler': self.lr_scheduler_actor.state_dict(),
            'critic_lr_scheduler': self.lr_scheduler_critic.state_dict()
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if 'actor_lr_scheduler' in checkpoint:
            self.lr_scheduler_actor.load_state_dict(checkpoint['actor_lr_scheduler'])
            self.lr_scheduler_critic.load_state_dict(checkpoint['critic_lr_scheduler'])

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
    
    # Define noise annealing schedule - faster decay
    noise_start = 0.8
    noise_end = 0.05
    noise_decay = NUM_EPISODES / 4  # Decay over quarter of the episodes
    
    # Early stopping variables
    patience = 100
    best_reward = -float('inf')
    no_improvement_count = 0
    
    # Training loop
    for episode in range(episode_start, NUM_EPISODES + episode_start):
        # Reset environment
        state, _ = env.reset()
        agent.noise.reset()
        
        # Set noise scale based on episode - exponential decay
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
                
                # Update learning rate schedulers
                if learning:
                    agent.lr_scheduler_actor.step(episode_reward)
                    agent.lr_scheduler_critic.step(episode_reward)
                
                # Save best model
                if episode_reward > max_reward: 
                    max_reward = episode_reward
                    if learning:
                        agent.save(f'models_{VERSION_NAME}/best_model.pth')
                
                # Early stopping logic
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                print(f"Episode {episode} finished after {t} steps with reward {episode_reward:.1f}")
                break
        
        # Check for early stopping
        if no_improvement_count >= patience:
            print(f"Early stopping triggered after {episode} episodes due to no improvement")
            break
        
        # Save more frequently at the beginning, then less often
        save_interval = REPORT_EPISODES if episode > 1000 else 50
        if learning and episode > 0 and episode % save_interval == 0:
            # Save model
            agent.save(f'models_{VERSION_NAME}/ddpg_model_{episode}.pth')
            print(f"Model saved at episode {episode}")
            
            # Plot progress every few save points to reduce overhead
            if episode % (save_interval * 2) == 0 or episode < 500:
                plt.figure(figsize=(15, 10))
                
                # Plot rewards
                plt.subplot(2, 2, 1)
                plt.plot(total_rewards[-300:])  # Show only recent rewards for clarity
                plt.ylabel('rewards')
                plt.xlabel('episode')
                plt.title(f'Recent Training rewards - episode {episode}')
                
                # Plot episode duration
                plt.subplot(2, 2, 2)
                plt.plot(episode_duration[-300:])  # Show only recent durations
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
                    
                    # Plot TD errors
                    plt.subplot(2, 2, 4)
                    plt.plot(agent.td_error_history[-1000:])
                    plt.ylabel('TD error')
                    plt.xlabel('optimization step')
                    plt.title('TD Error (last 1000 steps)')
                
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