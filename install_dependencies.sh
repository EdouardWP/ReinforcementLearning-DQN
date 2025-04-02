#!/bin/bash

# Install Python dependencies
pip install numpy>=1.24.0 matplotlib>=3.7.0 gymnasium>=0.29.0 torch>=2.0.0 pygame>=2.5.0
pip install stable-baselines3>=2.0.0 tensorboard>=2.10.0 tqdm>=4.64.0 cloudpickle>=2.2.0

echo "Dependencies installed successfully!"
echo ""
echo "To run the project:"
echo "1. For the original DDPG implementation: python Pyrace_RL_DQN_v3.py train"
echo "2. For the Stable-Baselines3 implementation: python Pyrace_SB3.py train --algo SAC"
echo ""
echo "After training, you can play with the trained model:"
echo "1. For DDPG: python Pyrace_RL_DQN_v3.py play"
echo "2. For SB3: python Pyrace_SB3.py play --algo SAC" 