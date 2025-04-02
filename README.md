# Reinforcement Learning DQN Project - Pit Lane Repairs

This project implements a Deep Reinforcement Learning solution for a car racing environment using both Deep Q-Networks (DQN) and Deep Deterministic Policy Gradient (DDPG) algorithms.

## Project Structure

- `Pyrace_RL_QTable.py`: Original implementation using Q-table
- `Pyrace_RL_DQN.py`: Implementation of Deep Q-Network (DQN)
- `Pyrace_RL_DQN_v3.py`: Implementation of Deep Deterministic Policy Gradient (DDPG) for continuous states and actions

## Environment Versions

- `Pyrace-v1`: Original discrete environment
- `Pyrace-v3`: Enhanced environment with continuous radar readings and continuous actions

## Improvements in v3

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
- Optimized network architecture for improved performance:
  - Leaky ReLU activations for better gradient flow
  - Reduced hidden layer sizes for faster computation
  - Improved weight initialization
- Adaptive exploration noise with annealing schedule
- Efficient batch learning with reduced rendering overhead
- Gradient clipping for training stability

## Performance Optimizations
- Selective rendering: only render at display intervals, not during training
- Prioritized experience replay for more efficient learning
- Batch processing optimizations and reduced network size
- Noise annealing schedule to gradually reduce exploration
- Smoothed learning curve with Huber Loss and weight decay
- Learn periodically instead of at every step to reduce computation
- Increased model saving frequency in early episodes

## How to Run

### Training
```bash
# Train from scratch
python Pyrace_RL_DQN_v3.py train

# Continue training from a specific episode
python Pyrace_RL_DQN_v3.py train_10000
```

### Testing
```bash
# Play with best model
python Pyrace_RL_DQN_v3.py play

# Play with a specific saved model
python Pyrace_RL_DQN_v3.py play_10000
```

## Requirements
See `requirements.txt` for dependencies.

## Acknowledgements
This project builds upon the "Car Driving" practice code, extending it to use Deep Reinforcement Learning techniques. 