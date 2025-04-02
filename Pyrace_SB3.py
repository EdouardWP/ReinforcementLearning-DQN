import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch

import gymnasium as gym
import gym_race

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# Choose algorithm: PPO, SAC, or TD3
ALGO_NAME = "SAC"  # Soft Actor-Critic is good for continuous action spaces
VERSION = "SB3_v1"
LOG_DIR = f"logs/{ALGO_NAME}_{VERSION}"
MODEL_DIR = f"models_{ALGO_NAME}_{VERSION}"

# Training parameters
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5
SAVE_FREQ = 10000

def make_env(rank, seed=0):
    """
    Create environment factory for Stable-Baselines3
    """
    def _init():
        env = gym.make("Pyrace-v3")
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    return _init

def train_agent(algo_name=ALGO_NAME, continue_training=False, checkpoint=None):
    """
    Train an agent using Stable-Baselines3
    """
    # Create directories
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([make_env(i) for i in range(1)])
    
    # Add normalization for better stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    
    # Callbacks for evaluation and checkpoints
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"{MODEL_DIR}/best/",
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=MODEL_DIR,
        name_prefix="sb3_model",
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    # Setup algorithm
    if continue_training and checkpoint:
        print(f"Loading model from {checkpoint}")
        if algo_name == "PPO":
            model = PPO.load(checkpoint, env=env)
        elif algo_name == "SAC":
            model = SAC.load(checkpoint, env=env)
        elif algo_name == "TD3":
            model = TD3.load(checkpoint, env=env)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    else:
        # Create new model with appropriate hyperparameters for each algorithm
        if algo_name == "PPO":
            model = PPO(
                "MlpPolicy", 
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=None,
                tensorboard_log=LOG_DIR,
                policy_kwargs=dict(
                    net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                    activation_fn=torch.nn.ReLU
                ),
                verbose=1
            )
        elif algo_name == "SAC":
            model = SAC(
                "MlpPolicy", 
                env,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=None,
                ent_coef='auto',
                target_update_interval=1,
                target_entropy='auto',
                use_sde=False,
                sde_sample_freq=-1,
                use_sde_at_warmup=False,
                tensorboard_log=LOG_DIR,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], qf=[256, 256]),
                    activation_fn=torch.nn.ReLU
                ),
                verbose=1
            )
        elif algo_name == "TD3":
            # Define action noise for exploration
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=0.1 * np.ones(n_actions)
            )
            
            model = TD3(
                "MlpPolicy", 
                env,
                learning_rate=3e-4,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "episode"),
                gradient_steps=1,
                action_noise=action_noise,
                policy_delay=2,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                tensorboard_log=LOG_DIR,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], qf=[256, 256]),
                    activation_fn=torch.nn.ReLU
                ),
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # Train the agent
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
        reset_num_timesteps=not continue_training
    )
    
    # Save final model
    final_model_path = f"{MODEL_DIR}/final_model_{algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(final_model_path)
    
    # Also save the normalization stats
    env.save(f"{MODEL_DIR}/vec_normalize_final.pkl")
    
    print(f"Training complete! Final model saved to {final_model_path}")
    return model, env

def play_agent(algo_name=ALGO_NAME, model_path=None):
    """
    Play with a trained agent
    """
    # If no model path specified, use the best model
    if model_path is None:
        model_path = f"{MODEL_DIR}/best/best_model"
    
    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip")
        return
    
    # Create environment
    env = gym.make("Pyrace-v3", render_mode="human")
    env = Monitor(env)
    
    # Load normalization stats if available
    vec_normalize_path = f"{MODEL_DIR}/vec_normalize_final.pkl"
    if os.path.exists(vec_normalize_path):
        env = VecNormalize.load(vec_normalize_path, env)
        # Don't update normalization statistics during evaluation
        env.training = False
        env.norm_reward = False
    
    # Load agent
    if algo_name == "PPO":
        model = PPO.load(model_path)
    elif algo_name == "SAC":
        model = SAC.load(model_path)
    elif algo_name == "TD3":
        model = TD3.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    print(f"Loaded model from {model_path}")
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10, deterministic=True, render=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Play one episode interactively
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        env.render()
    
    print(f"Episode finished with reward {total_reward:.2f}")
    env.close()

def compare_algorithms():
    """
    Train and compare multiple algorithms
    """
    algorithms = ["PPO", "SAC", "TD3"]
    rewards = {}
    
    for algo in algorithms:
        print(f"\n=== Training {algo} ===\n")
        model, env = train_agent(algo_name=algo)
        
        # Evaluate performance
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        rewards[algo] = (mean_reward, std_reward)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    algo_names = list(rewards.keys())
    means = [rewards[algo][0] for algo in algo_names]
    stds = [rewards[algo][1] for algo in algo_names]
    
    plt.bar(algo_names, means, yerr=stds, capsize=10)
    plt.ylabel('Mean Reward')
    plt.title('Performance Comparison of Different Algorithms')
    plt.savefig(f"{LOG_DIR}/algorithm_comparison.png")
    plt.close()
    
    # Print results
    print("\n=== Algorithm Comparison ===")
    for algo in algo_names:
        mean, std = rewards[algo]
        print(f"{algo}: {mean:.2f} ± {std:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or play with Stable-Baselines3 agents')
    parser.add_argument('mode', choices=['train', 'play', 'continue', 'compare'],
                       help='Mode: train a new agent, play with a trained agent, continue training, or compare algorithms')
    parser.add_argument('--algo', choices=['PPO', 'SAC', 'TD3'], default=ALGO_NAME,
                       help=f'Algorithm to use (default: {ALGO_NAME})')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved model for play or continue mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_agent(algo_name=args.algo)
    elif args.mode == 'play':
        play_agent(algo_name=args.algo, model_path=args.model)
    elif args.mode == 'continue':
        if args.model is None:
            print("Please specify a model to continue training")
        else:
            train_agent(algo_name=args.algo, continue_training=True, checkpoint=args.model)
    elif args.mode == 'compare':
        compare_algorithms() 