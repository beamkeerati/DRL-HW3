from __future__ import annotations
import numpy as np
import torch
import random
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            device=None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 1000,
            batch_size: int = 1,
    ) -> None:
        """
        Initialize the CartPole Agent with Linear Q-Learning.

        Args:
            device: The computation device (CPU/GPU)
            num_of_action (int): Number of discrete actions available.
            action_range (list): The range of continuous action values.
            n_observations (int): Dimensionality of the state space.
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float): The discount factor for future rewards.
            buffer_size (int): Size of the replay buffer.
            batch_size (int): Number of samples to use in each update.
        """        
        # Initialize the device for computation
        self.device = device
        self.n_observations = n_observations
        self.episode_durations = []
        
        # Initialize the weight matrix for linear function approximation
        # Each column represents weights for an action
        self.w = np.zeros((n_observations, num_of_action))
        
        # Use Xavier initialization for weights to improve convergence
        limit = 1 / np.sqrt(n_observations)
        self.w = np.random.uniform(-limit, limit, (n_observations, num_of_action))
        
        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        next_action: int,
        terminated: bool
    ):
        """
        Updates the weight vector using the Temporal Difference (TD) error 
        in Q-learning with linear function approximation.

        Args:
            obs: The current state observation.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs: The next state observation.
            next_action (int): The action taken in the next state (not used in Q-learning).
            terminated (bool): Whether the episode has ended.

        Returns:
            float: The TD error from this update.
        """
        # Extract observation as numpy array
        if isinstance(obs, dict):
            if "observation" in obs:
                obs = obs["observation"]
            elif "obs" in obs:
                obs = obs["obs"]
            else:
                # Fallback: use the first available value
                obs = next(iter(obs.values()))
                
        if isinstance(obs, torch.Tensor):
            obs_arr = obs.detach().cpu().numpy()
        else:
            obs_arr = np.array(obs)
            
        # Make sure obs_arr is 1D
        if obs_arr.ndim > 1:
            obs_arr = obs_arr.flatten()
        
        # Process next_obs similarly
        if isinstance(next_obs, dict):
            if "observation" in next_obs:
                next_obs = next_obs["observation"]
            elif "obs" in next_obs:
                next_obs = next_obs["obs"]
            else:
                next_obs = next(iter(next_obs.values()))
                
        if isinstance(next_obs, torch.Tensor):
            next_obs_arr = next_obs.detach().cpu().numpy()
        else:
            next_obs_arr = np.array(next_obs)
            
        # Make sure next_obs_arr is 1D
        if next_obs_arr.ndim > 1:
            next_obs_arr = next_obs_arr.flatten()
        
        # Calculate current Q-value for the action taken
        current_q = self.q(obs_arr, action)
        
        # Calculate target Q-value
        if terminated:
            target_q = reward
        else:
            # For Q-learning, use max Q-value for next state (off-policy)
            next_q_values = self.q(next_obs_arr)
            max_next_q = np.max(next_q_values)
            target_q = reward + self.discount_factor * max_next_q
        
        # Calculate TD error
        td_error = target_q - current_q
        
        # Update weights using TD error and gradient of Q with respect to weights
        # For linear approximation, the gradient is just the state features
        self.w[:, action] += self.lr * td_error * obs_arr
        
        # Return the TD error for monitoring
        return td_error

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state: The current state of the environment.
        
        Returns:
            torch.Tensor: The selected action as a tensor.
        """
        # Process state based on its type
        if isinstance(state, dict):
            if "observation" in state:
                state = state["observation"]
            elif "obs" in state:
                state = state["obs"]
            else:
                # Fallback: use the first available value
                state = next(iter(state.values()))
        
        # Debug state occasionally
        if random.random() < 0.01:  # print state 1% of the time
            print(f"State type: {type(state)}, Value: {state}")
        
        # Convert state to a numpy array
        if isinstance(state, torch.Tensor):
            state_arr = state.detach().cpu().numpy()
        else:
            state_arr = np.array(state)
            
        # Make sure state_arr is 1D
        if state_arr.ndim > 1:
            state_arr = state_arr.flatten()
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select random action
            action_index = random.randrange(self.num_of_action)
        else:
            # Exploit: select action with highest Q-value
            q_values = self.q(state_arr)
            action_index = np.argmax(q_values)
        
        # Decay epsilon after each action selection
        self.decay_epsilon()
        
        # Scale the discrete action index to a continuous action value
        return self.scale_action(action_index)

    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment in which the agent interacts.

        Returns:
            tuple: (total_reward, timestep, avg_td_error) - Statistics from this episode
        """
        # Initialize trajectory collection variables
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        timestep = 0
        td_errors = []
        
        while not done:
            # Predict action from the linear Q-function (epsilon-greedy)
            action_tensor = self.select_action(state)
            
            # Ensure action is properly formatted for environment
            action_cont = action_tensor
            if isinstance(action_tensor, torch.Tensor):
                if action_cont.dim() == 0:
                    action_cont = action_cont.view(1, 1)
                elif action_cont.dim() == 1:
                    action_cont = action_cont.unsqueeze(0)
            
            # Execute action in the environment and observe next state and reward
            next_state, reward, terminated, truncated, info = env.step(action_cont)
            done = terminated or truncated
            
            # Ensure reward is a scalar value
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
            
            # Add debug printing to track rewards occasionally
            if timestep % 20 == 0 or timestep < 5:
                print(f"  Step {timestep}: Action={action_tensor.item():.2f}, Reward={reward:.2f}")
            
            total_reward += reward
            
            # Convert continuous action back to its discrete index for update
            if self.num_of_action > 1:
                step_val = (self.action_range[1] - self.action_range[0]) / (self.num_of_action - 1)
                action_idx = int(round((action_tensor.item() - self.action_range[0]) / step_val)) if step_val != 0 else 0
            else:
                action_idx = 0
            
            # Update the weights using TD learning
            td_error = self.update(state, action_idx, reward, next_state, None, done)
            td_errors.append(td_error)
            
            # Store the experience in replay buffer (for potential batch updates)
            self.memory.add(state, action_idx, reward, next_state, done)
            
            # Update state for next step
            state = next_state
            
            timestep += 1
            if done:
                self.episode_durations.append(timestep)
                break
        
        # Calculate average TD error for this episode
        avg_td_error = sum(td_errors) / len(td_errors) if td_errors else 0
        
        # Return episode statistics - matching DQN interface
        return total_reward, timestep, avg_td_error