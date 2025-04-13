from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class MC(BaseAlgorithm):
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
        Initialize the Monte Carlo algorithm.

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
            control_type=ControlType.MONTE_CARLO,
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
        Update Q-values using Monte Carlo.

        This method accumulates the current transition (obs, action, reward, next_obs) into the episode history.
        When an episode terminates (done is True), it computes the return (G) for each time step in the episode
        (using discounted future rewards) and updates the Q-value for the corresponding state–action pair via:

            Q(s, a) ← Q(s, a) + α * (G - Q(s, a))

        After the update, the episode histories are cleared.

        Args:
            obs (dict): The current observation (state) as a dictionary.
            action (int): The discrete action taken in the current state.
            reward (float): The reward received after taking the action.
            next_obs (dict): The next observation (state) as a dictionary.
            done (bool): Flag indicating whether the episode has terminated.
        """
        # Append the current transition to the history.
        self.obs_hist.append(obs)
        self.action_hist.append(action)
        self.reward_hist.append(reward)
        
        # Only perform the Monte Carlo update if the episode has terminated.
        if done:
            G = 0.0
            # Iterate backwards over the episode history to compute returns.
            for t in reversed(range(len(self.reward_hist))):
                G = self.reward_hist[t] + self.discount_factor * G
                state = self.discretize_state(self.obs_hist[t])
                a = self.action_hist[t]
                # Increment the visit count for analysis (optional).
                self.n_values[state][a] += 1
                # Update the Q-value using the incremental update rule.
                self.q_values[state][a] += self.lr * (G - self.q_values[state][a])
            
            # Optionally, print the updated Q-values for debugging.
            # print(f"Q-value: {self.q_values}")
            
            # Clear the episode histories after the update.
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []

