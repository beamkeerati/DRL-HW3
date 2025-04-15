import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        """
        Actor network for policy approximation.

        Args:
            input_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            output_dim (int): Dimension of the action space.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Actor, self).__init__()

        # ========= put your code here ========= #
        # Define network layers without BatchNorm
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Initialize weights for better convergence
        self.init_weights()
        
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Xavier initialization for better convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for action selection.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Action probabilities.
        """
        # ========= put your code here ========= #
        # Process input state
        x = state
        
        # First layer with ReLU activation
        x = self.fc1(x)
        x = torch.relu(x)
        
        # Second layer with ReLU activation
        x = self.fc2(x)
        x = torch.relu(x)
        
        # Output layer with softmax for discrete action probabilities
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        
        return action_probs
        # ====================================== #

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, learning_rate=1e-4):
        """
        Critic network for value function approximation.

        Args:
            state_dim (int): Dimension of the state space.
            hidden_dim (int): Number of hidden units in layers.
            learning_rate (float, optional): Learning rate for optimization. Defaults to 1e-4.
        """
        super(Critic, self).__init__()

        # ========= put your code here ========= #
        # Define network layers for state-value estimation without BatchNorm
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)  # Output a single value estimate
        
        # Initialize weights
        self.init_weights()
        
        # Define optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # ====================================== #

    def init_weights(self):
        """
        Initialize network weights using Kaiming initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                nn.init.zeros_(m.bias)  # Initialize bias to 0

    def forward(self, state):
        """
        Forward pass for value estimation.

        Args:
            state (Tensor): Current state of the environment.

        Returns:
            Tensor: Estimated state value.
        """
        # ========= put your code here ========= #
        # Process input state
        x = state
        
        # First layer with ReLU activation
        x = self.fc1(x)
        x = torch.relu(x)
        
        # Second layer with ReLU activation
        x = self.fc2(x)
        x = torch.relu(x)
        
        # Output layer (no activation, raw value)
        value = self.fc3(x)
        
        return value
        # ====================================== #

class Actor_Critic(BaseAlgorithm):
    def __init__(self, 
                device = None, 
                num_of_action: int = 2,
                action_range: list = [-2.5, 2.5],
                n_observations: int = 4,
                hidden_dim = 256,
                dropout = 0.05, 
                learning_rate: float = 0.01,
                tau: float = 0.005,
                discount_factor: float = 0.95,
                buffer_size: int = 256,
                batch_size: int = 1,
                # PPO-specific parameters
                clip_ratio: float = 0.2,
                entropy_coef: float = 0.01,
                value_coef: float = 0.5,
                update_epochs: int = 4,
                gae_lambda: float = 0.95,
                ):
        """
        Proximal Policy Optimization (PPO) implementation.

        Args:
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_of_action (int, optional): Number of possible actions. Defaults to 2.
            action_range (list, optional): Range of action values. Defaults to [-2.5, 2.5].
            n_observations (int, optional): Number of observations in state. Defaults to 4.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to 256.
            learning_rate (float, optional): Learning rate. Defaults to 0.01.
            tau (float, optional): Soft update parameter. Defaults to 0.005.
            discount_factor (float, optional): Discount factor for future rewards. Defaults to 0.95.
            batch_size (int, optional): Size of training batches. Defaults to 1.
            buffer_size (int, optional): Replay buffer size. Defaults to 256.
            clip_ratio (float, optional): PPO clipping parameter. Defaults to 0.2.
            entropy_coef (float, optional): Entropy bonus coefficient. Defaults to 0.01.
            value_coef (float, optional): Value loss coefficient. Defaults to 0.5.
            update_epochs (int, optional): Number of update epochs per batch. Defaults to 4.
            gae_lambda (float, optional): GAE lambda parameter. Defaults to 0.95.
        """
        # Feel free to add or modify any of the initialized variables above.
        # ========= put your code here ========= #
        self.device = device
        self.actor = Actor(n_observations, hidden_dim, num_of_action, learning_rate).to(device)
        self.critic = Critic(n_observations, hidden_dim, learning_rate).to(device)
        
        # PPO specific parameters
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.update_epochs = update_epochs
        self.gae_lambda = gae_lambda
        
        # Training buffer variables
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminals = []
        
        self.batch_size = batch_size
        self.tau = tau
        self.discount_factor = discount_factor
        self.n_observations = n_observations
        self.num_of_action = num_of_action
        # ====================================== #

        super(Actor_Critic, self).__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size,
        )

    def select_action(self, state, training=True):
        """
        Selects an action based on the current policy with optional exploration noise.
        
        Args:
        state (Tensor): The current state of the environment.
        training (bool, optional): Whether the model is in training mode. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: 
                - action_idx: The selected action index.
                - log_prob: The log probability of the selected action.
                - scaled_action: The final action after scaling.
        """
        # ========= put your code here ========= #
        # Process state based on type
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
            
        # Convert to tensor if needed
        if not isinstance(state_tensor, torch.Tensor):
            state_tensor = torch.tensor(state_tensor, dtype=torch.float32, device=self.device)
            
        # Ensure state has batch dimension
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
            
        # Forward pass through actor network to get action probabilities
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action_idx = dist.sample()
        log_prob = dist.log_prob(action_idx)
        
        # Get value estimate from critic (useful for training later)
        value = self.critic(state_tensor) if training else None
        
        # Convert discrete action index to continuous action value
        scaled_action = self.scale_action(action_idx.item())
        
        if training:
            return action_idx, log_prob, scaled_action, value
        else:
            return scaled_action
        # ====================================== #
    
    def generate_sample(self, batch_size):
        """
        Generates a batch sample from memory for training.

        Returns:
            Tuple: A tuple containing:
                - state_batch (Tensor): The batch of current states.
                - action_batch (Tensor): The batch of actions taken.
                - reward_batch (Tensor): The batch of rewards received.
                - next_state_batch (Tensor): The batch of next states received.
                - done_batch (Tensor): The batch of dones received.
        """
        # Ensure there are enough samples in memory before proceeding
        # ========= put your code here ========= #
        # Sample a batch from memory
        batch = self.memory.sample()
        # ====================================== #
        
        # Sample a batch from memory
        # ========= put your code here ========= #
        if len(self.memory) < batch_size:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        # Process states properly
        processed_states = []
        for s in states:
            if isinstance(s, dict):
                if "observation" in s:
                    s_tensor = s["observation"]
                elif "obs" in s:
                    s_tensor = s["obs"]
                else:
                    # Fallback: use the first available value
                    s_tensor = next(iter(s.values()))
            else:
                s_tensor = s
                
            if not isinstance(s_tensor, torch.Tensor):
                s_tensor = torch.tensor(s_tensor, dtype=torch.float32, device=self.device)
            else:
                s_tensor = s_tensor.to(self.device)
                
            processed_states.append(s_tensor)
            
        # Process next_states similarly
        processed_next_states = []
        for s in next_states:
            if isinstance(s, dict):
                if "observation" in s:
                    s_tensor = s["observation"]
                elif "obs" in s:
                    s_tensor = s["obs"]
                else:
                    # Fallback: use the first available value
                    s_tensor = next(iter(s.values()))
            else:
                s_tensor = s
                
            if not isinstance(s_tensor, torch.Tensor):
                s_tensor = torch.tensor(s_tensor, dtype=torch.float32, device=self.device)
            else:
                s_tensor = s_tensor.to(self.device)
                
            processed_next_states.append(s_tensor)
            
        # Stack processed states and convert other data to tensors
        state_batch = torch.stack(processed_states)
        action_batch = torch.tensor(actions, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_state_batch = torch.stack(processed_next_states)
        done_batch = torch.tensor(dones, dtype=torch.bool, device=self.device)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        # ====================================== #

    def compute_gae(self, rewards, values, next_value, dones):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards (list): List of rewards
            values (list): List of value estimates
            next_value (float): Value estimate of the next state
            dones (list): List of terminal flags
            
        Returns:
            advantages (Tensor): Computed advantages
            returns (Tensor): Computed returns (advantages + values)
        """
        # First ensure everything is on CPU and converted to numpy safely
        # Process rewards - convert to NumPy array after ensuring they're CPU tensors
        rewards_np = []
        for r in rewards:
            if isinstance(r, torch.Tensor):
                # Move tensor to CPU if it's on GPU
                r_cpu = r.detach().cpu().item()
            else:
                r_cpu = r
            rewards_np.append(r_cpu)
        rewards = np.array(rewards_np, dtype=np.float32)
        
        # Process values - convert to NumPy array after ensuring they're CPU tensors
        values_np = []
        for v in values:
            if isinstance(v, torch.Tensor):
                # Move tensor to CPU if it's on GPU
                v_cpu = v.detach().cpu().item()
            else:
                v_cpu = v
            values_np.append(v_cpu)
        values = np.array(values_np, dtype=np.float32)
        
        # Process dones - convert to NumPy array
        dones_np = []
        for d in dones:
            if isinstance(d, torch.Tensor):
                # Move tensor to CPU if it's on GPU
                d_cpu = d.detach().cpu().item()
            else:
                d_cpu = d
            dones_np.append(d_cpu)
        dones = np.array(dones_np, dtype=np.float32)
        
        # Process next_value - ensure it's a scalar
        if isinstance(next_value, torch.Tensor):
            next_value = next_value.detach().cpu().item()
        
        # Initialize advantage array
        advantages = np.zeros_like(rewards, dtype=np.float32)
        
        # Append next value to values array for easier computation
        all_values = np.append(values, next_value)
        
        # GAE calculation
        gae = 0
        for t in reversed(range(len(rewards))):
            # Delta = reward + discount * next_value * (1 - done) - current_value
            delta = rewards[t] + self.discount_factor * all_values[t+1] * (1 - dones[t]) - all_values[t]
            # GAE = delta + discount * lambda * (1 - done) * previous_gae
            gae = delta + self.discount_factor * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # Calculate returns (advantage + value)
        returns = advantages + values
        
        # Convert to tensors
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages for stable training
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return advantages, returns

    def calculate_loss(self, states, actions, old_log_probs, advantages, returns):
        """
        Computes the PPO loss for policy optimization.

        Args:
            states (Tensor): The batch of current states.
            actions (Tensor): The batch of actions taken.
            old_log_probs (Tensor): Log probabilities of actions under old policy.
            advantages (Tensor): Computed advantages.
            returns (Tensor): Computed returns.

        Returns:
            Tensor: Computed actor & critic loss.
        """
        # ========= put your code here ========= #
        # Update Critic (Value Loss)
        value_preds = self.critic(states).view(-1)  # Explicitly reshape to 1D tensor

        value_loss = mse_loss(value_preds, returns)
        
        # Update Actor (Policy Loss)
        # Get current action probabilities and distribution
        action_probs = self.actor(states)
        dist = Categorical(action_probs)
        curr_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        # Calculate ratio of new and old policies
        ratio = torch.exp(curr_log_probs - old_log_probs)
        
        # Clipped objective function
        clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Entropy bonus to encourage exploration
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss combines policy and value losses
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        return total_loss, policy_loss, value_loss, entropy
        # ====================================== #

    def update_policy(self):
        """
        Update the policy using the calculated loss.

        Returns:
            float: Loss value after the update.
        """
        # ========= put your code here ========= #
        # Check if we have enough experiences
        if len(self.states) < self.batch_size:
            return None
            
        # Convert collected experiences to tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            next_value = self.critic(states[-1].unsqueeze(0)) if not self.terminals[-1] else torch.zeros(1, device=self.device)
            advantages, returns = self.compute_gae(self.rewards, self.values, next_value, self.terminals)
            old_log_probs = torch.stack(self.log_probs)
            
        # Multiple epochs of updates with the same data (a key feature of PPO)
        total_losses = []
        for _ in range(self.update_epochs):
            # Calculate new policy and value losses
            total_loss, policy_loss, value_loss, entropy = self.calculate_loss(
                states, actions, old_log_probs, advantages, returns
            )
            
            # Optimize actor network
            self.actor.optimizer.zero_grad()
            # Optimize critic network
            self.critic.optimizer.zero_grad()
            
            # Backpropagate combined loss
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            
            # Update networks
            self.actor.optimizer.step()
            self.critic.optimizer.step()
            
            total_losses.append(total_loss.item())
        
        # Clear experience buffer after update
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminals = []
        
        # Return average loss across all update epochs
        return sum(total_losses) / len(total_losses) if total_losses else 0.0
        # ====================================== #


    def update_target_networks(self, tau=None):
        """
        Perform soft update of target networks using Polyak averaging.

        Args:
            tau (float, optional): Update rate. Defaults to self.tau.
        """
        # ========= put your code here ========= #
        # PPO doesn't use target networks, so this function is a no-op
        pass
        # ====================================== #

    def learn(self, env, max_steps=1000, num_agents=1, noise_scale=0.1, noise_decay=0.99):
        """
        Train the agent on a single episode.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
            num_agents (int): Number of agents in the environment.
            noise_scale (float, optional): Initial exploration noise level. Defaults to 0.1.
            noise_decay (float, optional): Factor by which noise decreases per step. Defaults to 0.99.
            
        Returns:
            tuple: (total_reward, episode_length, loss) - Statistics from this episode.
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
        step_count = 0
        loss = 0.0
        # ====================================== #

        # Clear experience buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.terminals = []
        
        for step in range(max_steps):
            # Predict action from the policy network
            # ========= put your code here ========= #
            # Process state
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
                
            # Convert to tensor if needed
            if not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.tensor(state_tensor, dtype=torch.float32, device=self.device)
            else:
                state_tensor = state_tensor.clone().detach().to(self.device)
                
            # Ensure state has correct shape
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            # Select action using the current policy
            action_idx, log_prob, scaled_action, value = self.select_action(state)
            
            # Store experience for later training
            self.states.append(state_tensor)
            self.actions.append(action_idx.item())
            self.log_probs.append(log_prob)
            self.values.append(value)
            # ====================================== #

            # Execute action in the environment and observe next state and reward
            # ========= put your code here ========= #
            # Ensure action is properly formatted for the environment
            if isinstance(scaled_action, torch.Tensor):
                if scaled_action.dim() == 0:
                    scaled_action = scaled_action.view(1, 1)
                elif scaled_action.dim() == 1:
                    scaled_action = scaled_action.unsqueeze(0)
            
            # Execute the action
            next_state, reward, terminated, truncated, info = env.step(scaled_action)
            done = terminated or truncated
            # ====================================== #

            # Store the transition in memory
            # ========= put your code here ========= #
            # Store reward and terminal state
            # Make sure reward is a Python scalar, not a tensor
            if isinstance(reward, torch.Tensor):
                reward = reward.item()
                
            self.rewards.append(reward)
            self.terminals.append(done)
            
            # Accumulate reward for return calculation
            total_reward += reward
            
            # Store in replay buffer for potential additional learning
            self.memory.add(state, action_idx.item(), reward, next_state, done)
            
            # Update state for next step
            state = next_state
            step_count += 1
            # ====================================== #

            # End episode if done
            if done:
                break

        # Update policy after collecting a complete episode
        if len(self.states) > 0:
            loss = self.update_policy()
            
        # Return episode statistics in the format expected by train.py
        return total_reward, step_count, loss if loss is not None else 0.0