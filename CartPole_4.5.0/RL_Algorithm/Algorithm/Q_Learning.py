from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType


class Q_Learning(BaseAlgorithm):
    def __init__(
            self,
            num_of_action: int,
            action_range: list,
            discretize_state_weight: list,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float,
    ) -> None:
        """
        Initialize the Q-Learning algorithm.

        Args:
            num_of_action (int): Number of possible actions.
            action_range (list): Scaling factor for actions.
            discretize_state_weight (list): Scaling factor for discretizing states.
            learning_rate (float): Learning rate for Q-value updates.
            initial_epsilon (float): Initial value for epsilon in epsilon-greedy policy.
            epsilon_decay (float): Rate at which epsilon decays.
            final_epsilon (float): Minimum value for epsilon.
            discount_factor (float): Discount factor for future rewards.
        """
        super().__init__(
            control_type=ControlType.Q_LEARNING,
            num_of_action=num_of_action,
            action_range=action_range,
            discretize_state_weight=discretize_state_weight,
            learning_rate=learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
            discount_factor=discount_factor,
        )
        
    def update(self, obs: dict, action: int, reward: float, next_obs: dict, done: bool):
        """
        Update Q-values using Q-Learning.

        This method applies the Q-Learning update rule to improve policy decisions by updating the Q-table.
        The update rule is:
        
            Q(s, a) ← Q(s, a) + α * (r + γ * max_a' Q(s', a') - Q(s, a))
        
        where:
            - s: current state (discretized)
            - a: action taken
            - r: reward received
            - s': next state (discretized)
            - α: learning rate (self.lr)
            - γ: discount factor (self.discount_factor)
        
        Args:
            obs (dict): The current observation (state) as a dictionary.
            action (int): The discrete action taken.
            reward (float): The reward received after taking the action.
            next_obs (dict): The next observation (state) as a dictionary.
            done (bool): Flag indicating whether the episode has terminated.
        """
        # Discretize the current and next states using the provided discretization method.
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)

        # Retrieve the current Q-value for the state-action pair.
        current_q = self.q_values[state][action]
        
        
        # If the episode has terminated, no future reward is considered.
        if done:
            target = reward
        else:
            # Otherwise, compute the target as the sum of the immediate reward and the discounted maximum future reward.
            target = reward + self.discount_factor * np.max(self.q_values[next_state])
        
        # Update the Q-value using the learning rate.
        self.q_values[state][action] = current_q + self.lr * (target - current_q)

