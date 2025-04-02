# Reinforcement Learning DQN Project - Pit Lane Repairs

This project implements Deep Reinforcement Learning solutions for a car racing environment using advanced algorithms and optimization techniques.

## Project Structure

- `Pyrace_RL_QTable.py`: Original implementation using Q-table (Part 1 reference)
- `Pyrace_RL_DQN.py`: Implementation of Deep Q-Network (DQN) for Part 1
- `Pyrace_RL_DQN_v3.py`: Implementation of Deep Deterministic Policy Gradient (DDPG) for Part 2
- `Pyrace_SB3.py`: Implementation using Stable-Baselines3 library for Part 3 (Bonus)

## Environment Versions

- `Pyrace-v1`: Original discrete environment
- `Pyrace-v3`: Enhanced environment with continuous radar readings and continuous actions

## Improvements in v3 (Part 2)

### Continuous State Space
- Changed radar readings from discrete buckets to continuous normalized values (0-1)
- Provides more precise state information to the agent

### Continuous Action Space
- Added a 2-dimensional continuous action space:
  - Acceleration: -1 (brake) to 1 (accelerate)
  - Steering: -1 (turn right) to 1 (turn left)
- Allows for more nuanced control of the car

### Improved Reward Function
- Progressive rewards based on distance to checkpoint
- Speed-based rewards to encourage optimal driving speed
- Enhanced checkpoint rewards
- Added survival bonus to encourage longer episodes
- Reduced penalty for crashing to prevent excessive negative feedback

### Advanced Algorithm (DDPG)
- Implemented DDPG with optimizations for handling continuous action spaces
- Actor-Critic architecture with prioritized experience replay
- N-step returns for faster credit assignment
- Advanced exploration with adaptive noise annealing

## Further Optimizations (Highly Efficient Version)
- **Neural Network Architecture**:
  - ELU activation functions for better gradient flow
  - Improved network capacity with more hidden units
  - Layer normalization for critic stability
  - Conditional batch normalization for improved training
  
- **Learning Optimizations**:
  - N-step returns (multi-step learning) for faster credit assignment
  - Prioritized experience replay with importance sampling
  - Huber loss with importance sampling weights
  - Early stopping to prevent overtraining
  - Adaptive learning rate scheduling

- **Training Efficiency**:
  - Reduced episodes needed (convergence in ~500-1000 episodes vs. 2000+)
  - Much higher sample efficiency through better memory replay
  - Selective rendering only for evaluation episodes
  - Gaussian noise process (simpler than Ornstein-Uhlenbeck)
  - Noise annealing schedule with faster decay

## Stable-Baselines3 Implementation (Part 3 - Bonus)
- Implemented multiple state-of-the-art algorithms using Stable-Baselines3:
  - **SAC (Soft Actor-Critic)**: Off-policy algorithm optimized for continuous action spaces with entropy regularization
  - **PPO (Proximal Policy Optimization)**: On-policy algorithm with clipped objective for stable learning
  - **TD3 (Twin Delayed DDPG)**: Advanced version of DDPG with twin critics and delayed policy updates
  
- **Advanced Features**:
  - Vectorized environments for potential parallel training
  - Observation and reward normalization for stability
  - Automatic entropy coefficient tuning
  - Comprehensive logging and evaluation callbacks
  - Algorithm comparison utilities
  
- **Hyperparameter Optimization**:
  - Carefully tuned parameters for each algorithm
  - Larger network architectures for better representation
  - Custom-sized replay buffers and batch sizes

## Performance Comparison
- The optimized DDPG implementation converges in ~500-1000 episodes vs 2000+ for the original
- Stable-Baselines3 implementation (particularly SAC) achieves higher final performance
- Comparison of algorithms in terms of:
  - Sample efficiency (learning speed)
  - Final performance (maximum reward)
  - Training stability (consistency across runs)

## How to Run

### Training DDPG
```bash
# Train from scratch
python Pyrace_RL_DQN_v3.py train

# Continue training from a specific episode
python Pyrace_RL_DQN_v3.py train_10000
```

### Testing DDPG
```bash
# Play with best model
python Pyrace_RL_DQN_v3.py play

# Play with a specific saved model
python Pyrace_RL_DQN_v3.py play_10000
```

### Using Stable-Baselines3
```bash
# Train a new agent (default: SAC)
python Pyrace_SB3.py train --algo SAC

# Play with a trained agent
python Pyrace_SB3.py play --algo SAC --model path/to/model

# Continue training from a checkpoint
python Pyrace_SB3.py continue --algo SAC --model path/to/model

# Compare different algorithms (PPO, SAC, TD3)
python Pyrace_SB3.py compare
```

## Requirements
- Python 3.7+
- PyTorch
- Gymnasium
- Stable-Baselines3 (for Part 3)
- NumPy, Matplotlib
- pygame

## Acknowledgements
This project builds upon the "Car Driving" practice code, extending it to use Deep Reinforcement Learning techniques with a focus on efficiency and advanced algorithms. 