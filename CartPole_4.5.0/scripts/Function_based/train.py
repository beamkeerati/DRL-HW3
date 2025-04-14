"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
from datetime import datetime

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from RL_Algorithm.Function_based.DQN import DQN

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  # TensorBoard logging

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from datetime import datetime
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with function approximation RL algorithms."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # Get task and algorithm name
    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    algorithm_name = "DQN"  # Only DQN is implemented

    # Setup TensorBoard logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", task_name, algorithm_name, current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # hyperparameters
    num_of_action = 2                     # two discrete actions (e.g., push left or push right)
    action_range = [-2.5, 2.5]            # continuous force range corresponding to actions
    learning_rate = 1e-3                  # learning rate for optimizer
    hidden_dim = 64                       # number of neurons in the hidden layer
    n_episodes = 1000                     # number of training episodes
    initial_epsilon = 1.0                 # starting exploration rate
    epsilon_decay = 0.001                 # epsilon decay per step (or per action)
    final_epsilon = 0.001                 # minimum exploration rate
    discount = 0.95                       # discount factor for future rewards
    buffer_size = 10000                   # replay buffer capacity
    batch_size = 64                       # minibatch size for experience replay
    dropout = 0.2                         # dropout rate
    tau = 0.005                           # soft update parameter

    # Log hyperparameters to TensorBoard
    hp_dict = {
        "num_of_action": num_of_action,
        "action_range": str(action_range),
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "n_episodes": n_episodes,
        "initial_epsilon": initial_epsilon,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": final_epsilon,
        "discount_factor": discount,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "dropout": dropout,
        "tau": tau
    }
    
    for name, val in hp_dict.items():
        writer.add_text("hyperparameters", f"{name}: {val}", 0)

    # set up matplotlib
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print(f"Training with algorithm: {algorithm_name}")
    print(f"Device: {device}")

    # Create directory for saving model checkpoints
    save_dir = os.path.join(f"w/{task_name}", algorithm_name)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize DQN agent
    agent = DQN(
        device=device,
        num_of_action=num_of_action,
        action_range=action_range,
        n_observations=4,  # Cart-pole has 4 state variables
        hidden_dim=hidden_dim,
        dropout=dropout,
        learning_rate=learning_rate,
        tau=tau,
        initial_epsilon=initial_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount,
        buffer_size=buffer_size,
        batch_size=batch_size,
    )

    # Initialize statistics tracking
    episode_rewards = []
    episode_lengths = []
    losses = []
    epsilon_values = []
    q_value_stats = []

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    total_steps = 0
    start_time = time.time()
    
    # simulate environment
    while simulation_app.is_running():
        
        for episode in tqdm(range(n_episodes)):
            episode_start_time = time.time()
            
            # Train one episode with DQN
            returns = agent.learn(env)
            
            # Extract episode data (if available)
            episode_reward = 0
            episode_length = 0
            episode_loss = 0
            
            if returns is not None:
                # Some implementations might return episode statistics
                if isinstance(returns, tuple) and len(returns) >= 1:
                    episode_reward = returns[0]
                if isinstance(returns, tuple) and len(returns) >= 2:
                    episode_length = returns[1]
            
            # Update statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get current epsilon (exploration rate)
            epsilon_value = getattr(agent, 'epsilon', 0.0)
            epsilon_values.append(epsilon_value)
            
            # Calculate episode duration
            episode_duration = time.time() - episode_start_time
            
            # Calculate moving averages
            window_size = min(10, len(episode_rewards))
            avg_reward = sum(episode_rewards[-window_size:]) / window_size
            avg_length = sum(episode_lengths[-window_size:]) / window_size
            
            # Log to TensorBoard
            writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
            writer.add_scalar('Training/Episode_Length', episode_length, episode)
            writer.add_scalar('Training/Average_Reward', avg_reward, episode)
            writer.add_scalar('Training/Average_Length', avg_length, episode)
            writer.add_scalar('Training/Episode_Duration', episode_duration, episode)
            writer.add_scalar('Exploration/Epsilon', epsilon_value, episode)
            
            # Log model parameters periodically
            if episode % 50 == 0:
                if hasattr(agent, 'policy_net'):
                    for name, param in agent.policy_net.named_parameters():
                        writer.add_histogram(f'Parameters/{name}', param.data, episode)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, episode)
                
                # For DQN, record Q-value distribution
                if hasattr(agent, 'policy_net') and hasattr(agent, 'memory') and hasattr(agent, 'batch_size') and hasattr(agent, 'generate_sample'):
                    # Sample a batch if possible
                    if len(agent.memory) > agent.batch_size:
                        sample = agent.generate_sample(agent.batch_size)
                        if sample is not None and len(sample) >= 3:  # Make sure we have state_batch
                            _, _, state_batch, *_ = sample
                            with torch.no_grad():
                                q_values = agent.policy_net(state_batch)
                                q_means = q_values.mean(dim=0)
                                q_stds = q_values.std(dim=0)
                                
                                for action_idx in range(num_of_action):
                                    writer.add_scalar(f'Q_Values/Action_{action_idx}_Mean', q_means[action_idx], episode)
                                    writer.add_scalar(f'Q_Values/Action_{action_idx}_Std', q_stds[action_idx], episode)
                                
                                # Log Q-value distribution
                                writer.add_histogram('Q_Values/Distribution', q_values.flatten(), episode)
            
            # Print progress
            if episode % 10 == 0:
                print(f"\nEpisode {episode}/{n_episodes}")
                print(f"  Reward: {episode_reward:.2f} (Avg10: {avg_reward:.2f})")
                print(f"  Length: {episode_length} steps (Avg10: {avg_length:.1f})")
                print(f"  Epsilon: {epsilon_value:.4f}")
                print(f"  Duration: {episode_duration:.2f}s")
            
            # Save model periodically
            if episode % 100 == 0 or episode == n_episodes - 1:
                model_file = f"{algorithm_name}_ep{episode}.json"
                agent.save_w(save_dir, model_file)
                print(f"Model checkpoint saved: {model_file}")

        # Training completed
        training_time = time.time() - start_time
        
        # Log final performance metrics
        writer.add_scalar('Performance/Total_Training_Time_Minutes', training_time / 60, 0)
        writer.add_scalar('Performance/Episodes_Completed', n_episodes, 0)
        writer.add_scalar('Performance/Final_Average_Reward', avg_reward, 0)
        
        # Create and save final plots
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # Plot exploration parameters
        plt.subplot(2, 2, 3)
        plt.plot(epsilon_values)
        plt.title('Exploration Parameter (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        
        # Plot smoothed rewards
        plt.subplot(2, 2, 4)
        window_size = min(20, len(episode_rewards))
        smoothed_rewards = []
        for i in range(len(episode_rewards) - window_size + 1):
            smoothed_rewards.append(sum(episode_rewards[i:i+window_size]) / window_size)
        plt.plot(range(window_size-1, len(episode_rewards)), smoothed_rewards)
        plt.title(f'Smoothed Rewards (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'learning_curves.png'))
        
        # Save the final model
        final_model_file = f"{algorithm_name}_final.json"
        agent.save_w(save_dir, final_model_file)
        
        print("\n===== Training Complete =====")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Final model saved as: {final_model_file}")
        print(f"TensorBoard logs saved to: {log_dir}")
        print("\nTo view training metrics, run:")
        print(f"  tensorboard --logdir={log_dir}")
        
        writer.close()
        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()