"""Script to train RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Fix the import path - include all algorithms
from RL_Algorithm.Function_based.DQN import DQN
from RL_Algorithm.Function_based.Linear_Q import Linear_QN
from RL_Algorithm.Function_based.MC_REINFORCE import MC_REINFORCE

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default="DQN", choices=["DQN", "Linear_Q", "MC_REINFORCE"], 
                   help="Algorithm to use (DQN, Linear_Q, or MC_REINFORCE)")


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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters - MUST MATCH train.py values
    num_of_action = 3                     # three discrete actions
    action_range = [-2.5, 2.5]            # force range for CartPole
    learning_rate = 1e-3                  # moderate learning rate
    hidden_dim = 128                      # network capacity (DQN only)
    n_episodes = 5000                     # training episodes
    initial_epsilon = 1.0                 # start with full exploration
    epsilon_decay = 0.9985                # precisely calculated for 2000-step decay
    final_epsilon = 0.05                  # minimum exploration threshold
    discount = 0.99                       # discount factor
    buffer_size = 10000                   # experience buffer size
    batch_size = 64                       # batch size
    dropout = 0.2                         # dropout for regularization (DQN only)
    tau = 0.005                           # target network update rate (DQN only)

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

    # Get algorithm name
    algorithm_name = args_cli.algorithm
    print(f"Using algorithm: {algorithm_name}")

    # Initialize the appropriate agent based on algorithm choice
    if algorithm_name == "DQN":
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
    elif algorithm_name == "Linear_Q":
        agent = Linear_QN(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=4,  # Cart-pole has 4 state variables
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
    elif algorithm_name == "MC_REINFORCE":
        agent = MC_REINFORCE(
            device=device,
            num_of_action=num_of_action,
            action_range=action_range,
            n_observations=4,  # Cart-pole has 4 state variables
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            discount_factor=discount,
            # Include these for compatibility with BaseAlgorithm
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    episode = "final"  # Load the final model (or you can change to load a different episode)

    # Use EXACTLY the same naming convention as in train.py
    q_value_file = f"{algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.json"
    full_path = os.path.join(f"w/{task_name}", algorithm_name)

    print(f"Loading model from: {os.path.join(full_path, q_value_file)}")

    try:
        agent.load_w(full_path, q_value_file)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        # List available files to help debugging
        if os.path.exists(full_path):
            files = os.listdir(full_path)
            print(f"Available files in {full_path}:")
            for file in files:
                print(f"  - {file}")
        else:
            print(f"Directory {full_path} does not exist. Train a model first.")
            return

    # reset environment
    obs, _ = env.reset()
    timestep = 0

    # Statistics tracking
    episode_rewards = []
    episode_lengths = []

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0.0  # Initialize as float
                episode_length = 0

                while not done:
                    # Get action with minimal exploration
                    action_tensor = agent.select_action(obs)
                    action_cont = action_tensor.to(device)
                    
                    # Ensure action has correct shape
                    if action_cont.dim() == 0:
                        action_cont = action_cont.view(1, 1)
                    elif action_cont.dim() == 1:
                        action_cont = action_cont.unsqueeze(0)
                        
                    # Step the environment
                    next_obs, reward, terminated, truncated, _ = env.step(action_cont)
                    
                    # Convert tensor to float if needed
                    if isinstance(reward, torch.Tensor):
                        reward_value = reward.item()
                    else:
                        reward_value = float(reward)
                        
                    # Update statistics
                    episode_reward += reward_value
                    episode_length += 1
                    
                    # Check if done
                    done = terminated or truncated
                    obs = next_obs
                
                # Log episode results
                print(f"Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.4f}, Length = {episode_length}")
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
            
            # Display summary statistics
            if episode_rewards:
                # Convert all values to native Python floats
                float_rewards = [float(r) for r in episode_rewards]
                avg_reward = sum(float_rewards) / len(float_rewards)
                avg_length = sum(episode_lengths) / len(episode_lengths)
                
                print(f"\nAverage over {n_episodes} episodes:")
                print(f"  Average Reward: {avg_reward:.4f}")
                print(f"  Average Episode Length: {avg_length:.2f}")
                
                # Print algorithm-specific metrics
                if algorithm_name == "MC_REINFORCE":
                    print("  Algorithm: MC_REINFORCE (Policy Gradient)")
                elif algorithm_name == "DQN":
                    print("  Algorithm: DQN (Value-based)")
                elif algorithm_name == "Linear_Q":
                    print("  Algorithm: Linear Q-learning (Value-based)")
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        break
    # ==================================================================== #

    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()