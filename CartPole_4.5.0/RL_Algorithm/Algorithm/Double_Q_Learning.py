from __future__ import annotations
import numpy as np
from RL_Algorithm.RL_base import BaseAlgorithm, ControlType

class Double_Q_Learning(BaseAlgorithm):
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
        Initialize the Double Q-Learning algorithm.

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
            control_type=ControlType.DOUBLE_Q_LEARNING,
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
        Update Q-values using Double Q-Learning.

        This method applies the Double Q-Learning update rule:
        
            With 50% probability, update qa_values:
                a_max = argmax_a qa_values(next_state)
                target = r + γ * qb_values(next_state)[a_max]   (if not terminal)
            Otherwise, update qb_values:
                a_max = argmax_a qb_values(next_state)
                target = r + γ * qa_values(next_state)[a_max]   (if not terminal)
        
        Finally, update the overall q_values as the average of qa_values and qb_values.

        Args:
            obs (dict): The current observation (state).
            action (int): The discrete action taken.
            reward (float): The reward received after taking the action.
            next_obs (dict): The next observation (state).
            done (bool): Flag indicating whether the episode has terminated.
        """
        # Discretize the current and next states.
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)
        
        # Randomly decide which Q-table to update.
        if np.random.rand() < 0.5:
            # Update qa_values.
            current_q = self.qa_values[state][action]
            if done:
                target = reward
            else:
                a_max = int(np.argmax(self.qa_values[next_state]))
                target = reward + self.discount_factor * self.qb_values[next_state][a_max]
            self.qa_values[state][action] = current_q + self.lr * (target - current_q)
        else:
            # Update qb_values.
            current_q = self.qb_values[state][action]
            if done:
                target = reward
            else:
                a_max = int(np.argmax(self.qb_values[next_state]))
                target = reward + self.discount_factor * self.qa_values[next_state][a_max]
            self.qb_values[state][action] = current_q + self.lr * (target - current_q)
        
        # Update the overall Q-value as the average of qa_values and qb_values.
        self.q_values[state][action] = (self.qa_values[state][action] + self.qb_values[state][action]) / 2.0



