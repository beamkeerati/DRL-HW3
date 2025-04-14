from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base_function import BaseAlgorithm


class Linear_QN(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int = 2,
            action_range: list = [-2.5, 2.5],
            learning_rate: float = 0.01,
            initial_epsilon: float = 1.0,
            epsilon_decay: float = 1e-3,
            final_epsilon: float = 0.001,
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

        super().__init__(
            num_of_action=num_of_action,
            action_range=action_range,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
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
            obs (dict): The current state observation, containing feature representations.
            action (int): The action taken in the current state.
            reward (float): The reward received for taking the action.
            next_obs (dict): The next state observation.
            next_action (int): The action taken in the next state (used in SARSA).
            terminated (bool): Whether the episode has ended.

        """
        # ========= put your code here ========= #
        # Extract features from observation
        features = np.array(obs)
        
        # Calculate current Q-value
        current_q = self.q(obs, action)
        
        # Calculate target Q-value
        if terminated:
            target = reward
        else:
            # Q-learning uses the max Q-value of the next state (not the actual next action)
            next_q_values = self.q(next_obs)
            max_next_q = np.max(next_q_values)
            target = reward + self.discount_factor * max_next_q
        
        # Calculate TD error
        td_error = target - current_q
        
        # Update weights using gradient descent
        self.w[:, action] += self.lr * td_error * features
        # ====================================== #

    def select_action(self, state):
        """
        Select an action based on an epsilon-greedy policy.
        
        Args:
            state (Tensor): The current state of the environment.
        
        Returns:
            Tensor: The selected action.
        """
        # ========= put your code here ========= #
        # Convert state to feature representation
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Explore: random action
            action_idx = np.random.randint(self.num_of_action)
        else:
            # Exploit: best action
            q_values = self.q(state)
            action_idx = np.argmax(q_values)
        
        # Scale the action to continuous space
        action = self.scale_action(action_idx)
        
        return action, action_idx
        # ====================================== #

    def learn(self, env, max_steps):
        """
        Train the agent on a single step.

        Args:
            env: The environment in which the agent interacts.
            max_steps (int): Maximum number of steps per episode.
        """

        # ===== Initialize trajectory collection variables ===== #
        # Reset environment to get initial state (tensor)
        # Track total episode return (float)
        # Flag to indicate episode termination (boolean)
        # Step counter (int)
        # ========= put your code here ========= #
        # Reset environment
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        # SARSA algorithm loop
        prev_action = None
        
        while not done and steps < max_steps:
            # If this is the first step, select an action
            if prev_action is None:
                action, action_idx = self.select_action(state)
            else:
                action_idx = prev_action
                action = self.scale_action(action_idx)
            
            # Take a step in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Select next action
            next_action, next_action_idx = self.select_action(next_state)
            
            # Update weights (SARSA update)
            self.update(state, action_idx, reward.item(), next_state, next_action_idx, terminated)
            
            # Update tracking variables
            total_reward += reward.item()
            state = next_state
            prev_action = next_action_idx
            steps += 1
            
            # Decay epsilon
            self.decay_epsilon()
        # ====================================== #