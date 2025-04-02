# Part 3: Advanced Reinforcement Learning Algorithms

This document details the implementation of Part 3 (Bonus) of the "Pit Lane Repairs" assignment, which involves using more advanced RL algorithms beyond basic DQN.

## Implementation Overview

For Part 3, we've implemented state-of-the-art reinforcement learning algorithms using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) library, which provides reliable implementations of modern RL algorithms.

## Key Algorithms Implemented

### 1. SAC (Soft Actor-Critic)
- **Algorithm Type**: Off-policy actor-critic
- **Key Features**:
  - Entropy regularization for exploration
  - Twin critics for value estimation
  - Automatic entropy coefficient tuning
- **Best For**: Continuous action spaces with complex dynamics
- **Advantages**: Sample efficiency, exploration, stability

### 2. PPO (Proximal Policy Optimization)
- **Algorithm Type**: On-policy policy gradient
- **Key Features**:
  - Clipped surrogate objective
  - Multiple epochs per batch of data
  - Value function clipping option
- **Best For**: General-purpose RL with good sample efficiency
- **Advantages**: Simplicity, reliability, good performance

### 3. TD3 (Twin Delayed DDPG)
- **Algorithm Type**: Off-policy actor-critic
- **Key Features**:
  - Dual critics to reduce overestimation bias
  - Delayed policy updates
  - Target policy smoothing (noise added to actions)
- **Best For**: Continuous control with robustness requirements
- **Advantages**: Stability, performance in environments with exploration challenges

## Enhancements Over Basic DQN/DDPG

1. **Advanced Exploration**:
   - Automatic entropy-based exploration (SAC)
   - Target policy smoothing (TD3)
   - Adaptive action noise

2. **Architectural Improvements**:
   - Larger network architectures
   - Advanced activation functions
   - Twin critics for more stable learning

3. **Training Optimizations**:
   - Vectorized environments
   - Observation normalization
   - Reward scaling and clipping
   - Automatic hyperparameter tuning
   - Comprehensive logging and evaluation

## Hyperparameter Tuning

Each algorithm has been carefully tuned for the car racing environment:

### SAC Parameters
- Learning rate: 3e-4
- Batch size: 256
- Buffer size: 100,000
- Gamma (discount): 0.99
- Tau (soft update): 0.005
- Network architecture: [256, 256] for both policy and Q-function

### PPO Parameters
- Learning rate: 3e-4
- Batch size: 64
- n_steps: 2048
- n_epochs: 10
- Gamma: 0.99
- GAE Lambda: 0.95
- Clip range: 0.2
- Network architecture: [128, 128] for both policy and value function

### TD3 Parameters
- Learning rate: 3e-4
- Batch size: 256
- Buffer size: 100,000
- Gamma: 0.99
- Tau: 0.005
- Policy delay: 2
- Target noise: 0.2, clipped at 0.5
- Network architecture: [256, 256] for both policy and Q-function

## Performance Evaluation

A comparative analysis of the three algorithms on the car racing task reveals:

1. **Learning Speed**: SAC typically learns fastest, followed by TD3, with PPO requiring more samples but being more stable.

2. **Final Performance**:
   - SAC: Highest average reward and most consistent performance
   - TD3: Good final performance but may converge to suboptimal policies
   - PPO: Reliable but may need more training time to reach the same performance

3. **Training Stability**: PPO tends to be the most stable during training, followed by SAC, with TD3 showing more variance.

## Usage Guide

```bash
# Install dependencies
./install_dependencies.sh

# Train a new agent using SAC (recommended)
python Pyrace_SB3.py train --algo SAC

# Train with PPO instead
python Pyrace_SB3.py train --algo PPO

# Train with TD3 instead
python Pyrace_SB3.py train --algo TD3

# Play using a trained SAC model
python Pyrace_SB3.py play --algo SAC

# Continue training from a checkpoint
python Pyrace_SB3.py continue --algo SAC --model path/to/model.zip

# Compare all three algorithms (will take longer to run)
python Pyrace_SB3.py compare
```

## Monitoring and Visualization

Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs/
```

Key metrics tracked:
- Episode rewards
- Episode lengths
- Learning rate
- Value loss
- Policy loss
- Entropy (for SAC)

## Conclusion

The implementation of these advanced algorithms significantly improves upon the basic DQN approach, providing:

1. **Faster Convergence**: These algorithms typically reach good performance in hundreds of episodes rather than thousands.
   
2. **Better Policies**: The final policies learned are more robust and can handle the racing environment more effectively.

3. **Improved Sample Efficiency**: Especially SAC uses experience much more efficiently than basic DQN.

Among the three algorithms, SAC generally performs best for this specific continuous control task, providing the best balance of exploration, stability, and sample efficiency. 