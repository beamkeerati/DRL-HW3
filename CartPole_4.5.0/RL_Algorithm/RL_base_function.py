import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import os
import json
import torch
import torch.nn as nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """
        # ========= put your code here ========= #
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        # ====================================== #

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """
        # ========= put your code here ========= #
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        
        states = [item[0] for item in batch]
        actions = [item[1] for item in batch]
        rewards = [item[2] for item in batch]
        next_states = [item[3] for item in batch]
        dones = [item[4] for item in batch]
        
        return states, actions, rewards, next_states, dones
        # ====================================== #

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
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
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 1,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer(buffer_size, batch_size)
    
    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # ========= put your code here ========= #
        if isinstance(obs, dict):
            if 'policy' in obs:
                # Extract tensor and move to CPU, then flatten into a 1D array
                if isinstance(obs['policy'], torch.Tensor):
                    # Ensure tensor is moved to CPU before converting to NumPy
                    features = obs['policy'].detach().cpu().numpy().flatten()
                else:
                    features = np.array(obs['policy']).flatten()
            else:
                # Extract relevant observation features from the dictionary
                pose_cart = float(obs.get('pose_cart', 0.0))
                pose_pole = float(obs.get('pose_pole', 0.0))
                vel_cart = float(obs.get('vel_cart', 0.0))
                vel_pole = float(obs.get('vel_pole', 0.0))
                features = np.array([pose_cart, pose_pole, vel_cart, vel_pole])
        elif isinstance(obs, torch.Tensor):
            # Handle tensor observations - ensure it's moved to CPU first
            features = obs.detach().cpu().numpy().flatten()
        else:
            # If obs is not a dictionary, convert it to numpy array
            features = np.array(obs).flatten()

        if a is None:
            # Get q values from all actions in state
            q_values = np.zeros(self.num_of_action)
            for action in range(self.num_of_action):
                q_values[action] = np.dot(features, self.w[:, action])
            return q_values
        else:
            # Get q values given action & state
            return np.dot(features, self.w[:, a])
        # ====================================== #
        
    
    def scale_action(self, action):
        """
        Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

        Args:
            action (int): Discrete action in range [0, n].
            n (int): Number of discrete actions (inclusive range from 0 to n).
        
        Returns:
            torch.Tensor: Scaled action tensor.
        """
        # ========= put your code here ========= #
        action_min, action_max = self.action_range
        
        if self.num_of_action == 1:
            continuous_value = (action_min + action_max) / 2.0
        else:
            continuous_value = action_min + (action / (self.num_of_action - 1)) * (action_max - action_min)
        
        return torch.tensor([[continuous_value]], dtype=torch.float32)
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        
        model_params = {
            'weights': self.w.tolist()
        }
        
        with open(full_path, 'w') as f:
            json.dump(model_params, f)
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        full_path = os.path.join(path, filename)
        
        try:
            with open(full_path, 'r') as file:
                data = json.load(file)
                
            if 'weights' in data:
                self.w = np.array(data['weights'])
            else:
                raise KeyError("The loaded data does not contain weights.")
        except FileNotFoundError:
            print(f"Warning: File not found: {full_path}. Using default weights.")
        except json.JSONDecodeError:
            print(f"Warning: Error decoding JSON from file: {full_path}. Using default weights.")
        # ====================================== #