import numpy as np
from collections import defaultdict
from enum import Enum
import os
import json
import torch


class ControlType(Enum):
    """
    Enum representing different control algorithms.
    """
    MONTE_CARLO = 1
    TEMPORAL_DIFFERENCE = 2
    Q_LEARNING = 3
    DOUBLE_Q_LEARNING = 4
    SARSA = 5


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        control_type (ControlType): The type of control algorithm used.
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        control_type: ControlType,
        num_of_action: int,
        action_range: list,  # [min, max]
        discretize_state_weight: list,  # [pose_cart:int, pose_pole:int, vel_cart:int, vel_pole:int]
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float,
    ):
        self.control_type = control_type
        self.lr = learning_rate
        self.discount_factor = discount_factor 
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range
        self.discretize_state_weight = discretize_state_weight

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        # print(self.q_values)

        if self.control_type == ControlType.MONTE_CARLO:
            self.obs_hist = []
            self.action_hist = []
            self.reward_hist = []
        elif self.control_type == ControlType.DOUBLE_Q_LEARNING:
            self.qa_values = defaultdict(lambda: np.zeros(self.num_of_action))
            self.qb_values = defaultdict(lambda: np.zeros(self.num_of_action))

    def discretize_state(self, obs: dict):
        """
        Discretize the observation state by scaling continuous state values using the provided discretization weights.

        Args:
            obs (dict): Observation dictionary containing policy states.
                        Expected keys: 'pose_cart', 'pose_pole', 'vel_cart', 'vel_pole'.

        Returns:
            Tuple[int, int, int, int]: Discretized state as a tuple of four integers.
        """
        # Extract continuous state values from the observation dictionary
        if 'policy' in obs:
            # Extract tensor and move to CPU, then flatten into a 1D array
            policy = obs['policy'].detach().cpu().numpy().flatten()
            pose_cart, pose_pole, vel_cart, vel_pole = policy[0], policy[1], policy[2], policy[3]
        else:
            pose_cart = obs.get('pose_cart', 0.0)
            pose_pole = obs.get('pose_pole', 0.0)
            vel_cart = obs.get('vel_cart', 0.0)
            vel_pole = obs.get('vel_pole', 0.0)


        # Apply scaling factors from discretize_state_weight and round to the nearest integer
        disc_pose_cart = int(np.round(pose_cart * self.discretize_state_weight[0]))
        disc_pose_pole = int(np.round(pose_pole * self.discretize_state_weight[1]))
        disc_vel_cart = int(np.round(vel_cart * self.discretize_state_weight[2]))
        disc_vel_pole = int(np.round(vel_pole * self.discretize_state_weight[3]))

        # Return the discretized state as a tuple
        return (disc_pose_cart, disc_pose_pole, disc_vel_cart, disc_vel_pole)


    def get_discretize_action(self, obs_dis) -> int:
        """
        Select an action using an epsilon-greedy policy.

        Args:
            obs_dis (tuple): Discretized observation.

        Returns:
            int: Chosen discrete action index.
        """
        # With probability epsilon, explore by choosing a random action.
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_of_action)
        else:
            # Otherwise, choose the action with the highest estimated Q-value.
            q_vals = self.q_values[obs_dis]
            return int(np.argmax(q_vals))


    def mapping_action(self, action):
        """
        Maps a discrete action in range [0, n-1] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n-1].

        Returns:
            torch.Tensor: Scaled action tensor with shape (1, action_dim).
        """
        action_min, action_max = self.action_range
        # Handle the edge case where there's only one discrete action.
        if self.num_of_action == 1:
            continuous_value = (action_min + action_max) / 2.0
        else:
            # Linearly map the discrete action to a continuous value.
            continuous_value = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)
        # Return a 2D tensor with shape (1, 1)
        return torch.tensor([[continuous_value]], dtype=torch.float32)


    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        The epsilon value is multiplied by a decay factor but is not allowed to drop below final_epsilon.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def get_action(self, obs) -> torch.Tensor:
        """
        Get action based on epsilon-greedy policy.

        Args:
            obs (dict): The observation state.

        Returns:
            torch.Tensor, int: Scaled action tensor and chosen action index.
        """
        obs_dis = self.discretize_state(obs)
        action_idx = self.get_discretize_action(obs_dis)
        # print("action_idx: ", action_idx)
        action_tensor = self.mapping_action(action_idx)
        return action_tensor, action_idx


    def save_q_value(self, path, filename):
        """
        Save the model parameters to a JSON file.

        For DOUBLE_Q_LEARNING, compute the average Q-values from qa_values and qb_values,
        and store them along with the individual qa_values and qb_values.

        Args:
            path (str): Path to save the model.
            filename (str): Name of the file.
        """
        # Ensure the directory exists.
        os.makedirs(path, exist_ok=True)
        
        if self.control_type == ControlType.DOUBLE_Q_LEARNING:
            # Compute the average Q-values from qa_values and qb_values
            avg_q_values = {}
            for state in self.qa_values:
                avg = (self.qa_values[state] + self.qb_values[state]) / 2.0
                avg_q_values[state] = avg
            
            # Convert all dictionaries to have string keys and list values.
            avg_q_values_str_keys = {str(k): v.tolist() for k, v in avg_q_values.items()}
            qa_values_str_keys = {str(k): v.tolist() for k, v in self.qa_values.items()}
            qb_values_str_keys = {str(k): v.tolist() for k, v in self.qb_values.items()}
            
            model_params = {
                'q_values': avg_q_values_str_keys,
                'qa_values': qa_values_str_keys,
                'qb_values': qb_values_str_keys,
            }
            # print("model_params: ", model_params)
        elif self.control_type == ControlType.MONTE_CARLO:
            try:
                q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
            except:
                q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
            try:
                n_values_str_keys = {str(k): v.tolist() for k, v in self.n_values.items()}
            except:
                n_values_str_keys = {str(k): v for k, v in self.n_values.items()}
            
            model_params = {
                'q_values': q_values_str_keys,
                'n_values': n_values_str_keys,
            }
        else:
            try:
                q_values_str_keys = {str(k): v.tolist() for k, v in self.q_values.items()}
            except:
                q_values_str_keys = {str(k): v for k, v in self.q_values.items()}
            model_params = {
                'q_values': q_values_str_keys,
            }
            
        full_path = os.path.join(path, filename)
        with open(full_path, 'w') as f:
            json.dump(model_params, f)


            
    def load_q_value(self, path, filename):
        """
        Load model parameters from a JSON file.

        Args:
            path (str): Path where the model is stored.
            filename (str): Name of the file.

        Returns:
            dict: The loaded Q-values.
        """
        full_path = os.path.join(path, filename)
        try:
            with open(full_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {full_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {full_path}")

        if 'q_values' not in data:
            raise KeyError("The loaded data does not contain 'q_values'.")

        data_q_values = data['q_values']
        for state, action_values in data_q_values.items():
            state = state.replace('(', '').replace(')', '')
            tuple_state = tuple(map(float, state.split(', ')))
            self.q_values[tuple_state] = action_values.copy()
            if self.control_type == ControlType.DOUBLE_Q_LEARNING:
                self.qa_values[tuple_state] = action_values.copy()
                self.qb_values[tuple_state] = action_values.copy()

        if self.control_type == ControlType.MONTE_CARLO:
            if 'n_values' not in data:
                raise KeyError("The loaded data does not contain 'n_values' required for Monte Carlo.")
            data_n_values = data['n_values']
            for state, n_values in data_n_values.items():
                state = state.replace('(', '').replace(')', '')
                tuple_state = tuple(map(float, state.split(', ')))
                self.n_values[tuple_state] = n_values.copy()
        return self.q_values


