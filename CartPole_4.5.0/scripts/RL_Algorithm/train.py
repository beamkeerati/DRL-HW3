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

from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
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
parser.add_argument("--algorithm", type=str, default="Q_Learning", 
                    choices=["Q_Learning", "SARSA", "DOUBLE_Q_LEARNING", "MONTE_CARLO"],
                    help="RL algorithm to use for training")


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
import numpy as np
import matplotlib.pyplot as plt

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
    if algorithm_name == "Q_Learning":
        from RL_Algorithm.Algorithm.Q_Learning import Q_Learning
        return Q_Learning
    elif algorithm_name == "SARSA":
        from RL_Algorithm.Algorithm.SARSA import SARSA
        return SARSA
    elif algorithm_name == "DOUBLE_Q_LEARNING":
        from RL_Algorithm.Algorithm.Double_Q_Learning import Double_Q_Learning
        return Double_Q_Learning
    elif algorithm_name == "MONTE_CARLO":
        from RL_Algorithm.Algorithm.MC import MC
        return MC
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
    save_dir = os.path.join(f"q_value/{task_name}", algorithm_name)
    os.makedirs(save_dir, exist_ok=True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==================================================================== #
    # ========================= Can be modified ========================== #

    # hyperparameters
    num_of_action = 3
    action_range = [-1, 1.0]  # [min, max]
    discretize_state_weight = [10, 50, 10, 50]  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
    learning_rate = 0.05
    n_episodes = 500
    start_epsilon = 1.0
    epsilon_decay = 0.99  # reduce the exploration over time
    final_epsilon = 0.05
    discount = 0.99

    print(f"Training with algorithm: {algorithm_name}")
    print(f"Discretization weights: {discretize_state_weight}")
    print(f"Action space: {num_of_action} discrete actions mapped to range {action_range}")
    
    # Get the appropriate algorithm class
    AlgorithmClass = get_algorithm_class(algorithm_name)
    
    # Initialize agent
    agent = AlgorithmClass(
        num_of_action=num_of_action,
        action_range=action_range,
        discretize_state_weight=discretize_state_weight,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=discount
    )

    # Metrics for tracking performance
    episode_rewards = []
    episode_lengths = []
    q_value_changes = []
    visited_states = set()
    start_time = time.time()

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    total_steps = 0
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
        
            for episode in tqdm(range(n_episodes)):
                obs, _ = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode_start_time = time.time()

                # Store initial Q-values for calculating change
                if len(agent.q_values) > 0:
                    initial_q_values = {k: v.copy() for k, v in agent.q_values.items()}
                else:
                    initial_q_values = {}

                while not done:
                    # agent stepping
                    action, action_idx = agent.get_action(obs)

                    # env stepping
                    next_obs, reward, terminated, truncated, _ = env.step(action)

                    reward_value = reward.item()
                    terminated_value = terminated.item() 
                    episode_reward += reward_value
                    episode_step += 1
                    total_steps += 1

                    # Track visited states
                    state = agent.discretize_state(obs)
                    visited_states.add(state)

                    # Update the agent (algorithm specific)
                    if algorithm_name == "SARSA":
                        # For SARSA, we need to get the next action first
                        next_action_tensor, next_action_idx = agent.get_action(next_obs)
                        agent.update(
                            obs=obs,
                            action=action_idx,
                            reward=reward_value,
                            next_obs=next_obs,
                            next_action=next_action_idx,
                            done=terminated_value
                        )
                    else:
                        # For other algorithms (Q-Learning, Double Q-Learning, Monte Carlo)
                        agent.update(
                            obs=obs,
                            action=action_idx,
                            reward=reward_value,
                            next_obs=next_obs,
                            done=terminated_value
                        )

                    done = terminated or truncated
                    obs = next_obs
                
                # Calculate Q-value change for this episode
                if len(initial_q_values) > 0:
                    total_change = 0
                    count = 0
                    for state, values in agent.q_values.items():
                        if state in initial_q_values:
                            diff = np.sum(np.abs(values - initial_q_values[state]))
                            total_change += diff
                            count += 1
                    
                    avg_q_change = total_change / max(count, 1)
                    q_value_changes.append(avg_q_change)
                else:
                    q_value_changes.append(0)

                # Log metrics to TensorBoard
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                episode_time = time.time() - episode_start_time
                
                writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
                writer.add_scalar('Training/Episode_Length', episode_step, episode)
                writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
                writer.add_scalar('Training/Q_Value_Change', q_value_changes[-1], episode)
                writer.add_scalar('Training/Visited_States', len(visited_states), episode)
                writer.add_scalar('Training/Episode_Time', episode_time, episode)
                
                # Log average metrics every 10 episodes
                if episode % 10 == 0 and episode > 0:
                    avg_reward = sum(episode_rewards[-10:]) / 10
                    avg_length = sum(episode_lengths[-10:]) / 10
                    writer.add_scalar('Training/Avg_10_Episode_Reward', avg_reward, episode)
                    writer.add_scalar('Training/Avg_10_Episode_Length', avg_length, episode)
                    
                    print(f"Episode {episode}/{n_episodes}")
                    print(f"  Epsilon: {agent.epsilon:.4f}")
                    print(f"  Average Reward (last 10): {avg_reward:.4f}")
                    print(f"  Average Length (last 10): {avg_length:.1f}")
                    print(f"  Q-Value Change: {q_value_changes[-1]:.6f}")
                    print(f"  Visited States: {len(visited_states)}")
                    
                # Save checkpoints periodically
                if episode % 50 == 0 or episode == n_episodes - 1:
                    q_value_file = f"{algorithm_name}_{episode}_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
                    agent.save_q_value(save_dir, q_value_file)
                    print(f"Checkpoint saved: {q_value_file}")
                
                # Decay epsilon for exploration-exploitation balance
                agent.decay_epsilon()
             
        # Log final performance metrics
        training_time = time.time() - start_time
        writer.add_scalar('Performance/Total_Training_Time', training_time, 0)
        writer.add_scalar('Performance/Total_Steps', total_steps, 0)
        writer.add_scalar('Performance/Final_Visited_States', len(visited_states), 0)
        
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
        
        plt.subplot(2, 2, 3)
        plt.plot(q_value_changes)
        plt.title('Q-Value Changes')
        plt.xlabel('Episode')
        plt.ylabel('Average Change')
        
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
        final_model_file = f"{algorithm_name}_final_{num_of_action}_{action_range[1]}_{discretize_state_weight[0]}_{discretize_state_weight[1]}.json"
        agent.save_q_value(save_dir, final_model_file)
        
        print("!!! Training is complete !!!")
        print(f"Total training time: {training_time:.2f} seconds")
        print(f"Total steps: {total_steps}")
        print(f"Visited states: {len(visited_states)}")
        print(f"Final checkpoint saved: {final_model_file}")
        print(f"TensorBoard logs saved to: {log_dir}")
        print("\nTo view TensorBoard logs, run:")
        print(f"  python3 -m tensorboard.main --logdir={log_dir}")
        
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