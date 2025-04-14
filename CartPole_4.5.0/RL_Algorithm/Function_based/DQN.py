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
        self.fc1 = nn.Linear(n_observations, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, n_actions)
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
        x = x.float()                           # ensure input is float tensor
        x = F.relu(self.fc1(x))                 # hidden layer with ReLU activation
        x = self.dropout(x)                     # apply dropout for regularization
        x = self.fc2(x)                         # output layer (Q-values for each action)
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

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)

        self.episode_durations = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Experiment with different values and configurations to see how they affect the training process.
        # Remember to document any changes you make and analyze their impact on the agent's performance.

        pass
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

        # set up matplotlib
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        # If state is a dictionary, extract the observation using a common key.
        if isinstance(state, dict):
            if "observation" in state:
                state = state["observation"]
            elif "obs" in state:
                state = state["obs"]
            else:
                # Fallback: use the first available value.
                state = next(iter(state.values()))

        # Convert state to a torch tensor on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)  # add batch dimension if needed

        # Epsilon-greedy action selection: choose random action with probability epsilon
        if random.random() < self.epsilon:
            action_index = random.randrange(self.num_of_action)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state)
                action_index = int(torch.argmax(q_values).item())

        self.decay_epsilon()  # decay epsilon after each action selection

        # Scale the discrete action index to a continuous action value
        return self.scale_action(action_index)
        # ====================================== #

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
        # Compute Q(s, a) for current states using the policy network
        # (Gather the Q-values corresponding to the taken actions)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Compute the value of the next state using the target network (V(s') = max_a Q_target(s', a))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        if non_final_mask.any():
            with torch.no_grad():
                # Compute Q-values for non-final next states and take max over actions
                next_q_values = self.target_net(non_final_next_states)
                next_state_values[non_final_mask] = next_q_values.max(dim=1).values

        # Compute the target Q values: r + γ * max Q_target(s', a) (for non-final states; 0 for final states)
        target_q_values = reward_batch + (self.discount_factor * next_state_values)
        # Compute MSE loss between current Q-values and target Q-values
        loss = F.mse_loss(state_action_values, target_q_values)
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

        # Sample a batch of transitions from replay memory
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Helper function to process a state if it's a dictionary or tensor.
        def process_state(s):
            # If observation is a dict, try to extract the numeric observation
            if isinstance(s, dict):
                if "observation" in s:
                    s = s["observation"]
                elif "obs" in s:
                    s = s["obs"]
                else:
                    # Fallback: use the first available value from the dict
                    s = next(iter(s.values()))
            # If it's a torch tensor, move to CPU and convert to NumPy array
            if isinstance(s, torch.Tensor):
                s = s.cpu().detach().numpy()
            # Convert to a numpy array and ensure it is at least 1-dimensional
            s = np.array(s)
            if s.ndim == 0:
                s = np.array([s])
            return s

        # Process the list of states and next states
        processed_states = [process_state(s) for s in states]
        processed_next_states = [process_state(s) for s in next_states]

        # Force stacking into a 2D array so each sample is consistently shaped.
        state_batch = torch.tensor(np.vstack(processed_states), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Create a mask for non-final next states
        non_final_mask = torch.tensor([not d for d in dones], dtype=torch.bool, device=self.device)
        if non_final_mask.any():
            non_final_next_states = torch.tensor(
                np.vstack([s for s, d in zip(processed_next_states, dones) if not d]),
                dtype=torch.float32, device=self.device
            )
        else:
            non_final_next_states = torch.empty((0,), device=self.device)

        return non_final_mask, non_final_next_states, state_batch, action_batch, reward_batch

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
        self.optimizer.zero_grad()
        loss.backward()
        # (Optional) Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()
        return loss.item()
        # ====================================== #

    def update_target_networks(self):
        """
        Soft update of target network weights using Polyak averaging.
        """
        # Retrieve the state dictionaries (weights) of both networks
        # ========= put your code here ========= #
        policy_state_dict = self.policy_net.state_dict()
        target_state_dict = self.target_net.state_dict()
        # ====================================== #

        # Apply the soft update rule to each parameter in the target network
        # ========= put your code here ========= #
        for key in policy_state_dict:
            target_state_dict[key] = (1 - self.tau) * target_state_dict[key] + self.tau * policy_state_dict[key]
        # ====================================== #

        # Load the updated weights into the target network
        # ========= put your code here ========= #
        self.target_net.load_state_dict(target_state_dict)
        # ====================================== #

    def learn(self, env):
        """
        Train the agent on a single episode.

        Args:
            env: The environment to train in.
        """
        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        timestep = 0
        # ====================================== #

        while not done:
            # Predict action from the policy network (epsilon-greedy)
            # ========= put your code here ========= #
            action_tensor = self.select_action(state)
            # Ensure action_tensor is on the correct device
            action_cont = action_tensor.to(self.device)
            # Ensure that the action is a 2D tensor with shape [batch, action_dim].
            if action_cont.dim() == 0:
                action_cont = action_cont.view(1, 1)
            elif action_cont.dim() == 1:
                action_cont = action_cont.unsqueeze(0)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            next_state, reward, terminated, truncated, info = env.step(action_cont)
            done = terminated or truncated
            total_reward += reward
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            # Convert continuous action back to its discrete index for storage
            if self.num_of_action > 1:
                step_val = (self.action_range[1] - self.action_range[0]) / (self.num_of_action - 1)
                action_idx = int(round((action_tensor.item() - self.action_range[0]) / step_val)) if step_val != 0 else 0
            else:
                action_idx = 0
            self.memory.add(state, action_idx, reward, next_state, done)
            # ====================================== #

            # Update state for next step
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.update_policy()

            # Soft update of the target network's weights
            self.update_target_networks()

            timestep += 1
            if done:
                self.plot_durations(timestep)
                break



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