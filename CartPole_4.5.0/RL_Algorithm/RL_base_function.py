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
        self.memory.append((state, action, reward, next_state, done))

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
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

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
        # Convert observation to numpy array for computation
        if isinstance(obs, torch.Tensor):
            obs_arr = obs.detach().cpu().numpy()
        else:
            obs_arr = np.array(obs)
        if a is None:
            # Return Q-values for all actions (dot product of state and weight matrix)
            return obs_arr.dot(self.w)
        else:
            # Return Q-value for the given action index
            return float(obs_arr.dot(self.w[:, a]))
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
        action_min, action_max = self.action_range[0], self.action_range[1]
        if self.num_of_action == 1:
            # Only one action available: map to midpoint of range
            return torch.tensor((action_min + action_max) / 2.0, dtype=torch.float32)
        # Calculate step size for each discrete action
        step = (action_max - action_min) / (self.num_of_action - 1)
        return torch.tensor(action_min + action * step, dtype=torch.float32)
        # ====================================== #
    
    def decay_epsilon(self):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        if self.epsilon > self.final_epsilon:
            self.epsilon -= self.epsilon_decay
            if self.epsilon < self.final_epsilon:
                self.epsilon = self.final_epsilon
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        os.makedirs(path, exist_ok=True)
        full_path = os.path.join(path, filename)
        data = {}
        if hasattr(self, "policy_net"):
            # If using a neural network, save its state_dict (parameters)
            state_dict = self.policy_net.state_dict()
            # Convert tensor values to lists for JSON serialization
            for key, value in state_dict.items():
                data[key] = value.cpu().numpy().tolist()
        else:
            # If using linear weights, save the weight matrix
            data["w"] = self.w.tolist()
        # Write the data to a JSON file
        with open(full_path, 'w') as f:
            json.dump(data, f)
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weight parameters.
        """
        # ========= put your code here ========= #
        full_path = os.path.join(path, filename)
        with open(full_path, 'r') as f:
            data = json.load(f)
        if hasattr(self, "policy_net"):
            # Load parameters into the neural network
            state_dict = {}
            for key, value in data.items():
                state_dict[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
            self.policy_net.load_state_dict(state_dict)
            # If a target network exists, update it as well
            if hasattr(self, "target_net"):
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            # Load parameters into the linear weight matrix
            if "w" in data:
                self.w = np.array(data["w"])
        # ====================================== #


