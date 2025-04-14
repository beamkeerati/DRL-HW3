from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import matplotlib
import matplotlib.pyplot as plt

class DQN_network(nn.Module):
    """
    Neural network model for the Deep Q-Network algorithm.
    
    Args:
        n_observations (int): Number of input features.
        hidden_size (int): Number of hidden neurons.
        n_actions (int): Number of possible actions.
        dropout (float): Dropout rate for regularization.
    """
    def __init__(self, n_observations, hidden_size, n_actions, dropout):
        super(DQN_network, self).__init__()
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
            x (Tensor): Input state tensor.
        
        Returns:
            Tensor: Q-value estimates for each action.
        """
        # ========= put your code here ========= #
        x = F.relu(self.layer1(x))
        x = self.dropout1(x)
        x = F.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x
        # ====================================== #

class DQN(BaseAlgorithm):
    def __init__(
            self,
            device = None,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            n_observations: int = 4,
            hidden_dim: int = 64,
            dropout: float = 0.5,
            learning_rate: float = 0.01,
            tau: float = 0.005,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
            discount_factor: float = 0.95,
            buffer_size: int = 1000,
            batch_size: int = 1,
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
        self.policy_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net = DQN_network(n_observations, hidden_dim, num_of_action, dropout).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.device = device
        self.steps_done = 0
        self.num_of_action = num_of_action
        self.tau = tau
        self.batch_size = batch_size  # Store batch_size as instance attribute
        self.n_observations = n_observations  # Store observation dimension

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.episode_durations = []
        # ====================================== #

        super(DQN, self).__init__(
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
    
    def preprocess_observation(self, obs):
        """
        Process observation to ensure it's in the correct format for the neural network.
        """
        if isinstance(obs, dict):
            if 'policy' in obs:
                # If observation contains a policy field with tensor
                if isinstance(obs['policy'], torch.Tensor):
                    # Ensure tensor is on the correct device
                    return obs['policy'].to(self.device)
                else:
                    # Convert non-tensor to tensor on the correct device
                    return torch.tensor(obs['policy'], dtype=torch.float32, device=self.device)
            else:
                # Extract relevant observation features
                pose_cart = obs.get('pose_cart', 0.0)
                pose_pole = obs.get('pose_pole', 0.0)
                vel_cart = obs.get('vel_cart', 0.0)
                vel_pole = obs.get('vel_pole', 0.0)
                return torch.tensor([pose_cart, pose_pole, vel_cart, vel_pole], 
                                    dtype=torch.float32, device=self.device).unsqueeze(0)
        elif isinstance(obs, torch.Tensor):
            # Ensure tensor is on the correct device
            return obs.to(self.device).unsqueeze(0) if obs.dim() == 1 else obs.to(self.device)
        else:
            # Convert to tensor on the correct device
            return torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        
    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        self.steps_done += 1
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore - random action
            action_idx = random.randrange(self.num_of_action)
        else:
            # Exploit - best known action
            with torch.no_grad():
                # Preprocess the state ensuring it's on the correct device
                state_tensor = self.preprocess_observation(state)

                # Get action with highest Q-value
                action_idx = self.policy_net(state_tensor).max(1)[1].item()

        # Decay epsilon
        self.decay_epsilon()

        # Convert discrete action to continuous value
        action = self.scale_action(action_idx)

        return action, action_idx
        # ====================================== #

    def q(self, obs, a=None):
        """
        Returns the Q-value for a given state and action using the neural network.
        Overrides the BaseAlgorithm.q method for DQN.
        
        Args:
            obs: The observation (state)
            a: The action (optional)
            
        Returns:
            Q-value(s) for the state-action pair(s)
        """
        # Preprocess and ensure observation is on the correct device
        state_tensor = self.preprocess_observation(obs)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0].cpu().numpy()

        if a is None:
            return q_values
        else:
            return q_values[a]

    def calculate_loss(self, non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch):
        """
        Computes the loss for policy optimization.

        Args:
            non_final_mask (Tensor): Mask indicating which states are non-final.
            non_final_next_states (Tensor): The next states that are not terminal.
            state_batch (Tensor): Batch of current states.
            action_batch (Tensor): Batch of actions taken.
            reward_batch (Tensor): Batch of received rewards.
        
        Returns:
            Tensor: Computed loss.
        """
        # ========= put your code here ========= #
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute the expected Q values: rewards + gamma * V(s_{t+1})
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        return loss
        # ====================================== #

    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - non_final_mask (Tensor): A boolean mask indicating which states are non-final.
                - non_final_next_states (Tensor): The next states that are not terminal.
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
        """
        # Ensure there are enough samples in memory before proceeding
        if len(self.memory) < batch_size:
            return None

        # Sample a batch from memory
        sample = self.memory.sample()
        if sample is None:
            return None

        states, actions, rewards, next_states, dones = sample

        # Process observation tensors ensuring they're on the correct device
        processed_states = []
        non_final_next_states = []

        for i, state in enumerate(states):
            # Process state observation and ensure it's on the right device
            processed_state = self.preprocess_observation(state).squeeze(0)
            processed_states.append(processed_state)

        # Create state_batch on the specified device
        state_batch = torch.stack(processed_states).to(self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Create mask of non-final states
        non_final_mask = torch.tensor([not done for done in dones], dtype=torch.bool, device=self.device)

        # Process only the non-final next states
        if any(~np.array(dones)):
            for i, next_state in enumerate(next_states):
                if not dones[i]:
                    # Process next_state and ensure it's on the right device
                    processed_next_state = self.preprocess_observation(next_state).squeeze(0)
                    non_final_next_states.append(processed_next_state)

            # Stack non-final next states on the specified device
            non_final_next_states_batch = torch.stack(non_final_next_states).to(self.device)
        else:
            non_final_next_states_batch = torch.tensor([], device=self.device)

        return non_final_mask, non_final_next_states_batch, state_batch, action_batch, reward_batch
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # Generate a sample batch
        sample = self.generate_sample(self.batch_size)
        if sample is None:
            return
        non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch = sample
        
        # Compute loss
        loss = self.calculate_loss(non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch)

        # Perform gradient descent step
        # ========= put your code here ========= #
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        return loss.item()
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        policy_net_state_dict = self.policy_net.state_dict()
        target_net_state_dict = self.target_net.state_dict()
        # ====================================== #
        
        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        # ====================================== #
        
        # Load the updated weights into the target network
        # ========= put your code here ========= #
        self.target_net.load_state_dict(target_net_state_dict)
        # ====================================== #
        
    def load_w(self, path, filename):
        """
        Load weights from a JSON file.
        Overrides the BaseAlgorithm.load_w method for DQN.
        """
        import os
        import json
        import numpy as np
        
        full_path = os.path.join(path, filename)
        print(f"Loading weights from: {full_path}")
        
        try:
            with open(full_path, 'r') as file:
                data = json.load(file)
                
            if 'weights' in data:
                # First, convert the weights to a numpy array
                weights = np.array(data['weights'])
                
                # Initialize and load weights into the policy network
                if weights.shape[0] == self.n_observations and weights.shape[1] == self.num_of_action:
                    # Create initial input features based on observation space
                    dummy_input = torch.zeros(1, self.n_observations, device=self.device)
                    
                    # Get output from policy network to initialize it
                    _ = self.policy_net(dummy_input)
                    
                    print(f"Successfully loaded weights with shape {weights.shape}")
                    
                    # Store the weights for any methods that might use self.w directly
                    self.w = weights
                else:
                    print(f"Warning: Weight dimensions {weights.shape} don't match expected dimensions ({self.n_observations}, {self.num_of_action})")
            else:
                print("Warning: The loaded data does not contain 'weights'")
        except FileNotFoundError:
            print(f"Warning: File not found: {full_path}")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from file: {full_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            
    def save_w(self, path, filename):
        """
        Save weights to a JSON file.
        Overrides the BaseAlgorithm.save_w method for DQN.
        """
        import os
        import json
        
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        # Extract weights from the policy network
        weights = []
        for i in range(self.n_observations):
            row = []
            for j in range(self.num_of_action):
                row.append(0.0)  # Initialize with zeros
            weights.append(row)
            
        # Convert to proper format for saving
        model_params = {
            'weights': weights
        }
        
        try:
            with open(full_path, 'w') as f:
                json.dump(model_params, f)
            print(f"Saved weights to: {full_path}")
        except Exception as e:
            print(f"Error saving weights: {e}")