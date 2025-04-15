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
from RL_Algorithm.Function_based.Linear_Q import Linear_QN

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
parser.add_argument("--algorithm", type=str, default="DQN", choices=["DQN", "Linear_Q"], 
                    help="Algorithm to use (DQN or Linear_Q)")

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
    algorithm_name = args_cli.algorithm  # Use from command line arguments

    # Setup TensorBoard logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", task_name, algorithm_name, current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    # hyperparameters - EXACTLY MATCH WHAT PLAY.PY EXPECTS
    # Balanced exploration-exploitation hyperparameters
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
    
    # Log hyperparameters to TensorBoard
    hp_dict = {
        "algorithm": algorithm_name,
        "num_of_action": num_of_action,
        "action_range": str(action_range),
        "learning_rate": learning_rate,
        "n_episodes": n_episodes,
        "initial_epsilon": initial_epsilon,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": final_epsilon,
        "discount_factor": discount,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
    }
    
    if algorithm_name == "DQN":
        hp_dict.update({
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "tau": tau
        })
    
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
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

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
        
        # Training loop
        for episode in tqdm(range(n_episodes)):
            try:
                episode_start_time = time.time()

                # Print episode number at the beginning
                print(f"\nStarting Episode {episode}/{n_episodes}")

                # Train one episode with DQN
                returns = agent.learn(env)

                # Extract episode data with proper tensor handling
                episode_reward = to_scalar(returns[0]) if returns and len(returns) >= 1 else 0
                episode_length = to_scalar(returns[1]) if returns and len(returns) >= 2 else 0
                episode_loss = to_scalar(returns[2]) if returns and len(returns) >= 3 else 0

                # Debug print actual returned values
                print(f"  Raw returns: {returns}")
                print(f"  Processed: reward={episode_reward}, length={episode_length}, loss={episode_loss}")

                # Update statistics with scalar values
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                if episode_loss != 0:
                    losses.append(episode_loss)

                # Get current epsilon (exploration rate)
                epsilon_value = to_scalar(getattr(agent, 'epsilon', 0.0))
                epsilon_values.append(epsilon_value)

                # Calculate episode duration
                episode_duration = time.time() - episode_start_time

                # Calculate moving averages with error checking
                window_size = min(10, len(episode_rewards))
                if window_size > 0:
                    avg_reward = sum(episode_rewards[-window_size:]) / window_size
                    avg_length = sum(episode_lengths[-window_size:]) / window_size
                else:
                    avg_reward = 0
                    avg_length = 0

                # Log to TensorBoard
                writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
                writer.add_scalar('Training/Episode_Length', episode_length, episode)
                writer.add_scalar('Training/Average_Reward', avg_reward, episode)
                writer.add_scalar('Training/Average_Length', avg_length, episode)
                writer.add_scalar('Training/Episode_Duration', episode_duration, episode)
                writer.add_scalar('Exploration/Epsilon', epsilon_value, episode)

                if episode_loss != 0:
                    writer.add_scalar('Training/Loss', episode_loss, episode)

                # Print progress with type verification
                if episode % 10 == 0:
                    print(f"\nEpisode {episode}/{n_episodes}")
                    print(f"  Reward: {float(episode_reward):.2f} (Avg10: {float(avg_reward):.2f})")
                    print(f"  Length: {int(episode_length)} steps (Avg10: {float(avg_length):.1f})")
                    print(f"  Epsilon: {float(epsilon_value):.4f}")
                    print(f"  Duration: {float(episode_duration):.2f}s")

                # Save model periodically with EXACT naming format that play.py expects
                if episode % 100 == 0 or episode == n_episodes - 1:
                    model_file = f"{algorithm_name}_{episode}_{num_of_action}_{action_range[1]}.json"
                    agent.save_w(save_dir, model_file)
                    print(f"Model checkpoint saved: {model_file}")

                # Always save a checkpoint at episode 0 (this is what play.py expects by default)
                if episode == 0:
                    model_file = f"{algorithm_name}_0_{num_of_action}_{action_range[1]}.json"
                    agent.save_w(save_dir, model_file)
                    print(f"Initial model saved: {model_file}")

            except Exception as e:
                print(f"Error during episode {episode}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with next episode instead of crashing the whole training
                continue

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
        final_model_file = f"{algorithm_name}_final_{num_of_action}_{action_range[1]}.json"
        agent.save_w(save_dir, final_model_file)
        
        print("\n===== Training Complete =====")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"TensorBoard logs saved to: {log_dir}")
        print("\nTo view training metrics, run:")
        print(f"  tensorboard --logdir={log_dir}")
        
        writer.close()
        break
    # ==================================================================== #

    # close the simulator
    env.close()
    
def to_scalar(value):
    """
    Convert a value to a Python scalar if it's a tensor.

    Args:
        value: Any value, potentially a tensor

    Returns:
        The value as a Python scalar (int, float, etc.)
    """
    if isinstance(value, torch.Tensor):
        # Handle different tensor shapes/types
        if value.numel() == 1:  # Single element tensor
            return value.item()
        else:  # Multi-element tensor (unexpected in this context)
            return value.detach().cpu().numpy().tolist()
    return value

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()