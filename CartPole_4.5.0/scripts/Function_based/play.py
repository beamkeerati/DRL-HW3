"""Script to play RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm

# add argparse arguments
parser = argparse.ArgumentParser(description="Play with a trained RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--algorithm", type=str, default="DQN", 
                    choices=["DQN", "Linear_Q", "AC", "MC_REINFORCE"],
                    help="RL algorithm to use for play")
parser.add_argument("--model_path", type=str, help="Path to the model file")
parser.add_argument("--n_episodes", type=int, default=10, help="Number of episodes to play")


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
import numpy as np
from collections import deque

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
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

@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with trained agent."""
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

    # hyperparameters
    num_of_action = 3
    action_range = [-1, 1.0]  # [min, max]
    # For IsaacLab environments, we need to hardcode the observation dimension
    # CartPole has 4 observation dimensions: cart position, cart velocity, pole angle, pole velocity
    n_observations = 4  # Hardcoded for CartPole environment
    hidden_dim = 64
    dropout = 0.3
    learning_rate = 0.01
    tau = 0.005
    n_episodes = args_cli.n_episodes
    initial_epsilon = 0.01  # Low epsilon for evaluation
    epsilon_decay = 1.0     # No decay during evaluation
    final_epsilon = 0.01
    discount_factor = 0.99
    buffer_size = 10000
    batch_size = 64

    algorithm_name = args_cli.algorithm
    print(f"Playing with algorithm: {algorithm_name}")
    
    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    print("device: ", device)

    # Get the appropriate algorithm class
    AlgorithmClass = get_algorithm_class(algorithm_name)
    
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

    # Determine model path and load the model
    task_name = str(args_cli.task).split('-')[0]  # Stabilize, SwingUp
    
    if args_cli.model_path:
        model_path = args_cli.model_path
        if os.path.isfile(model_path):
            model_dir, model_file = os.path.split(model_path)
        else:
            model_dir = model_path
            # Find the latest model file in the directory
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{algorithm_name}_") and f.endswith(".json")]
            if not model_files:
                raise ValueError(f"No model files found in {model_dir}")
            model_files.sort()
            model_file = model_files[-1]  # Get the last file (assuming naming convention with episode numbers)
    else:
        model_dir = os.path.join(f"w/{task_name}", algorithm_name)
        # Find the latest model file in the directory
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f"{algorithm_name}_") and f.endswith(".json")]
            if model_files:
                model_files.sort()
                model_file = model_files[-1]
            else:
                raise ValueError(f"No model files found in {model_dir}")
        else:
            raise ValueError(f"Model directory {model_dir} not found")
    
    # Load the model
    full_path = os.path.join(model_dir, model_file)
    print(f"Loading model from: {full_path}")
    
    if algorithm_name in ["DQN", "Linear_Q"]:
        agent.load_w(model_dir, model_file)
    # For neural network models like Actor-Critic and REINFORCE, loading would require implementing 
    # specific load methods in those classes

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    
    # For collecting performance data
    episode_rewards = []
    episode_lengths = []
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            for episode in range(n_episodes):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                episode_steps = 0
                print(f"Episode {episode+1}/{n_episodes}")

                while not done:
                    # agent stepping (adjusting for different action selection methods)
                    if algorithm_name == "AC":  # Actor-Critic has a different action selection method
                        action, _ = agent.select_action(obs, noise=-0.1)  # Use negative noise as a flag for evaluation
                    else:
                        action, action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    # Update tracking variables
                    episode_reward += reward.item()
                    episode_steps += 1
                    
                    # Print progress
                    if episode_steps % 10 == 0:
                        print(f"  Step: {episode_steps}, Reward: {episode_reward:.2f}")
                    
                    done = terminated or truncated
                    obs = next_obs
                
                # Record episode statistics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_steps)
                print(f"Episode {episode+1} finished with reward {episode_reward:.2f} in {episode_steps} steps")
        
            # Print summary statistics
            print("\nPlay Summary:")
            print(f"Average Reward: {np.mean(episode_rewards):.2f}±{np.std(episode_rewards):.2f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.2f}±{np.std(episode_lengths):.2f}")
            print(f"Min/Max Reward: {min(episode_rewards):.2f}/{max(episode_rewards):.2f}")
            
            # Plot results
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.plot(episode_lengths)
            plt.title('Episode Lengths')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the figure
            os.makedirs("play_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            plt.savefig(f"play_results/{algorithm_name}_{task_name}_{timestamp}.png")
            plt.show()
            
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