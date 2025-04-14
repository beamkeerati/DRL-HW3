"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
from datetime import datetime
import subprocess
import importlib.util

# Check if tensorboard is installed, if not install it
def ensure_tensorboard_installed():
    try:
        import tensorboard
        print("TensorBoard is already installed.")
    except ImportError:
        print("TensorBoard not found, installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
        print("TensorBoard installed successfully!")

# Install TensorBoard if needed
ensure_tensorboard_installed()

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

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
parser.add_argument("--algorithm", type=str, default="DQN", 
                    choices=["DQN", "Linear_Q", "AC", "MC_REINFORCE"],
                    help="RL algorithm to use for training")
parser.add_argument("--n_episodes", type=int, default=500, help="Number of episodes for training")


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
import random

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
# from omni.isaac.lab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Dynamic import of the selected algorithm
def get_algorithm_class(algorithm_name):
    if algorithm_name == "DQN":
        from RL_Algorithm.Function_based.DQN import DQN
        return DQN
    elif algorithm_name == "Linear_Q":
        from RL_Algorithm.Function_based.Linear_Q import Linear_QN
        return Linear_QN
    elif algorithm_name == "AC":
        from RL_Algorithm.Function_based.AC import Actor_Critic
        return Actor_Critic
    elif algorithm_name == "MC_REINFORCE":
        from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE
        return MC_REINFORCE
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

# Import extensions to set up environment tasks
import CartPole.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

steps_done = 0

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with stable-baselines agent."""
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # directory for logging into
    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    algorithm_name = args_cli.algorithm
    
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "tensorboard", task_name, algorithm_name, current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Create directory for saving model checkpoints
    save_dir = os.path.join(f"w/{task_name}", algorithm_name)
    os.makedirs(save_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 3
    action_range = [-1, 1.0]  # [min, max]
    # For IsaacLab environments, we need to determine the observation dimension differently
    # CartPole has 4 observation dimensions: cart position, cart velocity, pole angle, pole velocity
    n_observations = 4  # Hardcoded for CartPole environment
    hidden_dim = 64
    dropout = 0.3
    learning_rate = 0.01
    tau = 0.005
    n_episodes = args_cli.n_episodes
    initial_epsilon = 1.0
    epsilon_decay = 0.995
    final_epsilon = 0.01
    discount_factor = 0.99
    buffer_size = 10000
    batch_size = 64

    print(f"Training with algorithm: {algorithm_name}")
    print(f"Action space: {num_of_action} discrete actions mapped to range {action_range}")
    print(f"Episodes: {n_episodes}")
    
    # Get the appropriate algorithm class
    AlgorithmClass = get_algorithm_class(algorithm_name)
    
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

    print("device: ", device)

    # Initialize agent based on selected algorithm
    if algorithm_name == "DQN":
        agent = AlgorithmClass(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            tau=tau,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
    elif algorithm_name == "MC_REINFORCE":
        agent = AlgorithmClass(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
        )
    elif algorithm_name == "Linear_Q":
        agent = AlgorithmClass(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
    elif algorithm_name == "AC":  # Actor-Critic (PPO)
        agent = AlgorithmClass(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=n_observations,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            tau=tau,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    # Try to load the model if it exists - otherwise start fresh
    try:
        # Find the latest model file in the directory
        model_files = [f for f in os.listdir(save_dir) if f.startswith(f"{algorithm_name}_") and f.endswith(".json")]
        if model_files:
            model_files.sort()
            model_file = model_files[-1]  # Get the last file (assuming naming convention with episode numbers)
            print(f"Loading existing model from: {os.path.join(save_dir, model_file)}")
            if algorithm_name in ["DQN", "Linear_Q"]:
                agent.load_w(save_dir, model_file)
        else:
            print(f"No existing model found in {save_dir}. Starting with a new model.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Starting with a new model: {e}")

    # Metrics for tracking performance
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    visited_states = set()
    start_time = time.time()

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    total_steps = 0
    
    # simulate environment
    try:
        while simulation_app.is_running():
            for episode in tqdm(range(n_episodes)):
                episode_start_time = time.time()
                
                # Different learning function calls based on algorithm
                if algorithm_name == "Linear_Q":
                    reward, steps = agent.learn(env, max_steps=500)
                    episode_reward = reward
                    episode_steps = steps
                elif algorithm_name == "AC":  # Actor-Critic (PPO)
                    agent.learn(env, max_steps=500, num_agents=1)
                    # Since PPO agent doesn't directly return rewards/steps, we'll evaluate after training
                    obs, _ = env.reset()
                    done = False
                    episode_reward = 0
                    episode_steps = 0
                    while not done and episode_steps < 500:
                        action, _ = agent.select_action(obs, noise=-0.1)  # Use negative noise as a flag for evaluation
                        next_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        episode_reward += reward.item()
                        episode_steps += 1
                        obs = next_obs
                elif algorithm_name == "MC_REINFORCE":
                    episode_reward, loss, _ = agent.learn(env)
                    episode_steps = agent.episode_durations[-1] if agent.episode_durations else 0
                    training_losses.append(loss)
                elif algorithm_name == "DQN":  # DQN
                    episode_reward, episode_steps = 0, 0
                    obs, _ = env.reset()
                    done = False
                    timestep = 0
                    
                    while not done and timestep < 500:
                        # Select action
                        action, action_idx = agent.select_action(obs)
                        
                        # Take action
                        next_obs, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        reward_value = reward.item()
                        
                        # Store transition
                        agent.memory.add(obs, action_idx, reward_value, next_obs, done)
                        
                        # Update policy
                        loss = agent.update_policy()
                        if loss is not None:
                            training_losses.append(loss)
                        
                        # Update target networks
                        agent.update_target_networks()
                        
                        # Update state and counters
                        obs = next_obs
                        episode_reward += reward_value
                        timestep += 1
                        
                    # After the episode ends, update counts
                    episode_steps = timestep
                
                # Log metrics to TensorBoard
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                episode_time = time.time() - episode_start_time
                total_steps += episode_steps
                
                writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
                writer.add_scalar('Training/Episode_Length', episode_steps, episode)
                if hasattr(agent, 'epsilon'):
                    writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
                if training_losses and episode < len(training_losses):
                    writer.add_scalar('Training/Loss', training_losses[episode], episode)
                writer.add_scalar('Training/Episode_Time', episode_time, episode)
                
                # Log average metrics every 10 episodes
                if episode % 10 == 0 and episode > 0:
                    avg_reward = sum(episode_rewards[-10:]) / 10
                    avg_length = sum(episode_lengths[-10:]) / 10
                    writer.add_scalar('Training/Avg_10_Episode_Reward', avg_reward, episode)
                    writer.add_scalar('Training/Avg_10_Episode_Length', avg_length, episode)
                    
                    print(f"Episode {episode}/{n_episodes}")
                    if hasattr(agent, 'epsilon'):
                        print(f"  Epsilon: {agent.epsilon:.4f}")
                    print(f"  Average Reward (last 10): {avg_reward:.4f}")
                    print(f"  Average Length (last 10): {avg_length:.1f}")
                    
                # Save checkpoints periodically
                if episode % 50 == 0 or episode == n_episodes - 1:
                    if algorithm_name in ["DQN", "Linear_Q"]:
                        w_file = f"{algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.json"
                        agent.save_w(save_dir, w_file)
                        print(f"Checkpoint saved: {w_file}")
            
            # Log final performance metrics
            training_time = time.time() - start_time
            writer.add_scalar('Performance/Total_Training_Time', training_time, 0)
            writer.add_scalar('Performance/Total_Steps', total_steps, 0)
            
            # Plot and save learning curves
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(2, 2, 2)
            plt.plot(episode_lengths)
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            
            if training_losses:
                plt.subplot(2, 2, 3)
                plt.plot(training_losses)
                plt.title('Training Losses')
                plt.xlabel('Episode')
                plt.ylabel('Loss')
            
            plt.subplot(2, 2, 4)
            # Plot smoothed reward
            window_size = min(10, len(episode_rewards))
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
            if algorithm_name in ["DQN", "Linear_Q"]:
                final_model_file = f"{algorithm_name}_final_{num_of_action}_{action_range[1]}.json"
                agent.save_w(save_dir, final_model_file)
            
            print("!!! Training is complete !!!")
            print(f"Total training time: {training_time:.2f} seconds")
            print(f"Total steps: {total_steps}")
            if algorithm_name in ["DQN", "Linear_Q"]:
                print(f"Final checkpoint saved: {final_model_file}")
            print(f"TensorBoard logs saved to: {log_dir}")
            print("\nTo view TensorBoard logs, run:")
            print(f"  python3 -m tensorboard.main --logdir={log_dir}")
            
            writer.close()
            break
    except Exception as e:
        print(f"Training interrupted: {e}")
        # Save the model even if training was interrupted
        if algorithm_name in ["DQN", "Linear_Q"]:
            interrupted_model_file = f"{algorithm_name}_interrupted_{num_of_action}_{action_range[1]}.json"
            agent.save_w(save_dir, interrupted_model_file)
            print(f"Interrupted checkpoint saved: {interrupted_model_file}")
    
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()