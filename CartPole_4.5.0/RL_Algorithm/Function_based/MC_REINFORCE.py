from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


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
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.layer3 = nn.Linear(hidden_size, n_actions)
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
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return F.softmax(x, dim=-1)
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
    ) -> None:
        """
        Initialize the CartPole Agent.

        Args:
            learning_rate (float): The learning rate for updating Q-values.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decays over time.
            final_epsilon (float): The final exploration rate.
            discount_factor (float, optional): The discount factor for future rewards. Defaults to 0.95.
        """     

        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.LR = learning_rate

        self.policy_net = MC_REINFORCE_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate)

        self.device = device
        self.steps_done = 0

        self.episode_durations = []
        # ====================================== #

        super(MC_REINFORCE, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
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
        # Calculate discounted returns
        returns = []
        G = 0
        
        # Calculate returns in reverse order
        for r in reversed(rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)  # Insert at the beginning
            
        # Convert to tensor
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        return returns
        # ====================================== #
    
    def generate_trajectory(self, env):
        """
        Generate a trajectory by interacting with the environment.

        Args:
            env: The environment object.
        
        Returns:
            Tuple: (episode_return, stepwise_returns, log_prob_actions, trajectory)
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Store state-action-reward history (list)
        # Store log probabilities of actions (list)
        # Store rewards at each step (list)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        # Reset environment
        state, _ = env.reset()
        
        # Initialize trajectory variables
        trajectory = []
        log_prob_actions = []
        rewards = []
        episode_return = 0
        done = False
        timestep = 0
        # ====================================== #
        
        # ===== Collect trajectory through agent-environment interaction ===== #
        while not done:
            
            # Predict action from the policy network
            # ========= put your code here ========= #
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities from policy network
            action_probs = self.policy_net(state_tensor)
            
            # Create a distribution and sample action
            dist = distributions.Categorical(action_probs)
            action_tensor = dist.sample()
            
            # Get log probability of the action
            log_prob = dist.log_prob(action_tensor)
            
            # Convert to action and scale
            action_idx = action_tensor.item()
            action = self.scale_action(action_idx)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_value = reward.item()
            # ====================================== #

            # Store action log probability reward and trajectory history
            # ========= put your code here ========= #
            log_prob_actions.append(log_prob)
            rewards.append(reward_value)
            trajectory.append((state, action_idx, reward_value, next_state, done))
            
            # Update tracking variables
            episode_return += reward_value
            state = next_state
            timestep += 1
            # ====================================== #

            if done:
                self.plot_durations(timestep)
                break

        # ===== Stack log_prob_actions &  stepwise_returns ===== #
        # ========= put your code here ========= #
        # Calculate stepwise returns
        stepwise_returns = self.calculate_stepwise_returns(rewards)
        
        # Stack log probabilities
        log_prob_actions = torch.cat([lp.unsqueeze(0) for lp in log_prob_actions])
        
        return episode_return, stepwise_returns, log_prob_actions, trajectory
        # ====================================== #
    
    def calculate_loss(self, stepwise_returns, log_prob_actions):
        """
        Compute the loss for policy optimization.

        Args:
            stepwise_returns (Tensor): Stepwise returns for the trajectory.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        # Policy gradient loss: -log_prob * return
        policy_gradient = -log_prob_actions * stepwise_returns
        
        # Mean loss over the trajectory
        loss = policy_gradient.mean()
        
        return loss
        # ====================================== #

    def update_policy(self, stepwise_returns, log_prob_actions):
        """
        Update the policy using the calculated loss.

        Args:
            stepwise_returns (Tensor): Stepwise returns.
            log_prob_actions (Tensor): Log probabilities of actions taken.
        
        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        # Calculate loss
        loss = self.calculate_loss(stepwise_returns, log_prob_actions)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        # ====================================== #
    
    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        
        Returns:
            Tuple: (episode_return, loss, trajectory)
        """
        # ========= put your code here ========= #
        self.policy_net.train()
        episode_return, stepwise_returns, log_prob_actions, trajectory = self.generate_trajectory(env)
        loss = self.update_policy(stepwise_returns, log_prob_actions)
        return episode_return, loss, trajectory
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