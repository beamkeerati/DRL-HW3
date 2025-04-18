from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm  # Changed import to match other algorithms

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt

class MC_REINFORCE_network(nn.Module):
    """
    Neural network for the MC_REINFORCE algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """

    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(MC_REINFORCE_network, self).__init__()
        # ========= put your code here ========= #
        # Create a network similar to DQN but with softmax output for action probabilities
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Batch normalization for better training
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Second hidden layer
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, n_actions)  # Output layer for action logits
        
        # Initialize weights with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        # ====================================== #

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor representing action probabilities.
        """
        # ========= put your code here ========= #
        x = x.float()  # ensure input is float tensor
        
        # First hidden layer
        x = self.fc1(x)
        if x.size(0) > 1:  # Apply batch norm only when batch size > 1
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer - softmax for action probabilities
        x = self.fc3(x)
        action_probs = F.softmax(x, dim=-1)
        
        return action_probs
        # ====================================== #

class MC_REINFORCE(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            discount_factor: float = 0.95,
            initial_epsilon: float = 0.0,  # Not used but added for compatibility 
            epsilon_decay: float = 1.0,    # Not used but added for compatibility
            final_epsilon: float = 0.0,    # Not used but added for compatibility
            buffer_size: int = 1000,       # Not used but added for compatibility
            batch_size: int = 1,           # Not used but added for compatibility
    ) -> None:
        """
        Initialize the CartPole Agent with REINFORCE algorithm.

        Args:
            device: The computation device (CPU/GPU)
            num_of_action (int): Number of discrete actions available.
            action_range (list): The range of continuous action values.
            n_observations (int): Dimensionality of the state space.
            hidden_dim (int): Size of hidden layers in the policy network.
            dropout (float): Dropout rate for regularization.
            learning_rate (float): The learning rate for policy optimization.
            discount_factor (float): The discount factor for future rewards.
            initial_epsilon, epsilon_decay, final_epsilon: Not used in REINFORCE but 
                             included for compatibility with other algorithms.
            buffer_size, batch_size: Not used in REINFORCE but included for compatibility.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.steps_done = 0
        self.episode_durations = []

        # Debug flags
        self.debug_mode = True  # Set to True to print detailed debugging information
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,  # Not used but needed for compatibility
            epsilon_decay=epsilon_decay,      # Not used but needed for compatibility
            final_epsilon=final_epsilon,      # Not used but needed for compatibility
            discount_factor=discount_factor,
            buffer_size=buffer_size,          # Not used but needed for compatibility
            batch_size=batch_size,            # Not used but needed for compatibility
        )

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()
    
    def calculate_stepwise_returns(self, rewards):
        """
        Compute stepwise returns for the trajectory.

        Args:
            rewards (list): List of rewards obtained in the episode.
        
        Returns:
            Tensor: Normalized stepwise returns.
        """
        # ========= put your code here ========= #
        # Initialize list to hold returns
        returns = []
        R = 0
        
        # Compute discounted returns for each step (backwards)
        for r in reversed(rewards):
            R = r + self.discount_factor * R
            returns.insert(0, R)
        
        # Convert returns to tensor
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize returns for stable training (optional but recommended)
        if len(returns) > 1 and returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        if self.debug_mode:
            print(f"Returns shape: {returns.shape}, mean: {returns.mean().item():.4f}, std: {returns.std().item():.4f}")
            print(f"Returns range: min={returns.min().item():.4f}, max={returns.max().item():.4f}")
        
        return returns
        # ====================================== #

    def select_action(self, state):
        """
        Select an action using the current policy.

        Args:
            state: The current state of the environment.

        Returns:
            torch.Tensor: The selected action tensor
        """
        if isinstance(state, dict):
            if "observation" in state:
                state_tensor = state["observation"]
            elif "obs" in state:
                state_tensor = state["obs"]
            else:
                # Fallback: use the first available value
                state_tensor = next(iter(state.values()))
        else:
            state_tensor = state

        # Convert state to tensor if needed
        if not isinstance(state_tensor, torch.Tensor):
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32, device=self.device)

        # Ensure state is the right shape (add batch dimension if needed)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Get action probabilities from policy network
        action_probs = self.policy_net(state_tensor)

        # Create a distribution and sample an action
        dist = distributions.Categorical(action_probs)
        action_idx = dist.sample()

        # Convert discrete action index to continuous action value
        # Map from discrete action index to continuous value in action_range
        action_range = self.action_range
        num_actions = self.num_of_action

        # Calculate normalized position in range based on action index
        normalized_pos = action_idx.item() / (num_actions - 1)

        # Linear mapping to action range
        action_value = action_range[0] + normalized_pos * (action_range[1] - action_range[0])

        # Create tensor for action value
        action_tensor = torch.tensor([action_value], device=self.device)

        # Return tuple of (action_idx, placeholder, action_tensor)
        return action_idx, None, action_tensor

    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (episode_return, saved_states, saved_actions, rewards, trajectory_length)
        """
        # Reset environment to get initial state
        state, _ = env.reset()
        
        # Initialize lists to store trajectory information
        trajectory = []  # (state, action, reward, next_state, done)
        saved_states = []  # Store states for later policy evaluation
        saved_actions = []  # Store actions for later policy evaluation
        rewards = []  # Store rewards at each step
        
        total_reward = 0.0  # Track total episode return
        done = False  # Flag for episode termination
        timestep = 0  # Step counter
        
        # Collect trajectory through agent-environment interaction
        while not done:
            # Process state based on its type
            if isinstance(state, dict):
                if "observation" in state:
                    state_tensor = state["observation"]
                elif "obs" in state:
                    state_tensor = state["obs"]
                else:
                    # Fallback: use the first available value
                    state_tensor = next(iter(state.values()))
            else:
                state_tensor = state
                
            # Convert state to tensor if needed
            if not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32, device=self.device)
                
            # Ensure state is the right shape (add batch dimension if needed)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            # Save the state for later policy evaluation
            saved_states.append(state_tensor)
            
            # Select action - this doesn't need gradients since we'll recalculate later
            with torch.no_grad():
                action_idx, _, action_tensor = self.select_action(state_tensor)
            
            # Save the action for later policy evaluation
            saved_actions.append(action_idx)
            
            # Ensure action is properly formatted for environment
            action_cont = action_tensor
            if isinstance(action_tensor, torch.Tensor):
                if action_cont.dim() == 0:
                    action_cont = action_cont.view(1, 1)
                elif action_cont.dim() == 1:
                    action_cont = action_cont.unsqueeze(0)
                    
            # Execute action in environment and observe next state and reward
            next_state, reward, terminated, truncated, info = env.step(action_cont)
            done = terminated or truncated
            
            # Ensure reward is a scalar value
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
                
            # Debug output occasionally
            if self.debug_mode and (timestep % 20 == 0 or timestep < 5):
                print(f"  Step {timestep}: Action={action_tensor.item():.2f}, Reward={reward:.2f}")
            
            # Store trajectory information
            trajectory.append((state, action_idx.item(), reward, next_state, done))
            rewards.append(reward)
            total_reward += reward
            
            # Update state for next step
            state = next_state
            
            timestep += 1
            if done:
                self.plot_durations(timestep)
                break
                
        # Calculate stepwise returns
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        
        if self.debug_mode:
            print(f"Trajectory length: {len(trajectory)}, Total reward: {total_reward:.2f}")
            print(f"States: {len(saved_states)}, Actions: {len(saved_actions)}, Returns: {len(stepwise_returns)}")
        
        return total_reward, saved_states, saved_actions, stepwise_returns, timestep
    
    def calculate_loss(self, saved_states, saved_actions, stepwise_returns):
        """
        Compute the loss for policy optimization with proper gradient tracking.

        Args:
            saved_states (list): List of states from the trajectory.
            saved_actions (list): List of actions taken.
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
        
        Returns:
            Tensor: Computed loss.
        """
        # Recalculate log probabilities with gradients enabled
        policy_losses = []
        for state_tensor, action_idx, R in zip(saved_states, saved_actions, stepwise_returns):
            # Get action probabilities from the policy network
            action_probs = self.policy_net(state_tensor)
            
            # Create a distribution
            dist = distributions.Categorical(action_probs)
            
            # Get the log probability of the taken action
            log_prob = dist.log_prob(action_idx)
            
            # Calculate policy gradient loss: -log(π(a|s)) * R
            # Note: We negate because we want to maximize expected return
            policy_losses.append(-log_prob * R)
        
        # Sum all policy losses
        policy_loss = torch.stack(policy_losses).sum()
        
        if self.debug_mode:
            print(f"Policy loss: {policy_loss.item():.4f}")
            
            # Check for exploding/vanishing gradients
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                print("WARNING: Loss is NaN or Inf!")
        
        return policy_loss

    def update_policy(self, saved_states, saved_actions, stepwise_returns):
        """
        Update the policy using the calculated loss.

        Args:
            saved_states (list): List of states from the trajectory.
            saved_actions (list): List of actions taken.
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
        
        Returns:
            float: Loss value after the update.
        """
        # Set policy network to training mode
        self.policy_net.train()
        
        # Calculate loss with proper gradient tracking
        loss = self.calculate_loss(saved_states, saved_actions, stepwise_returns)
        
        # Handle problematic loss values
        if torch.isnan(loss) or torch.isinf(loss):
            print("WARNING: Loss is NaN or Inf! Skipping update.")
            return 0.0
        
        # Perform gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Return the loss as a Python float
        return loss.item()
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (total_reward, timestep, loss) - Statistics from this episode
                   IMPORTANT: This matches the return format expected by train.py
        """
        # ========= put your code here ========= #
        # Generate trajectory and collect experience
        try:
            # Set to eval mode for trajectory collection
            self.policy_net.eval()
            episode_return, saved_states, saved_actions, stepwise_returns, timestep = self.generate_trajectory(env)
            
            # Set to train mode for policy update
            self.policy_net.train()
            # Compute the loss and update the policy
            loss = self.update_policy(saved_states, saved_actions, stepwise_returns)
            
            # Print debug info
            if self.debug_mode:
                print(f"REINFORCE learn returns: reward={episode_return:.2f}, length={timestep}, loss={loss:.4f}")
            
            # Return values in the exact format expected by train.py:
            # (total_reward, timestep, loss)
            return float(episode_return), int(timestep), float(loss)
        
        except Exception as e:
            print(f"Error in REINFORCE learn: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return safe values in case of error
            return 0.0, 0, 0.0
        # ====================================== #


    # Consider modifying this function to visualize other aspects of the training process.
    # ================================================================================== #
    def plot_durations(self, timestep=None, show_result=False):
        if timestep is not None:
            self.episode_durations.append(timestep)

        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
    # ================================================================================== #