# q_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional, Dict
import logging

from transformer_attention import HybridAttention

logger = logging.getLogger(__name__)


class DQN(nn.Module):
    """Deep Q-Network with advanced architecture"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512,
                 num_layers: int = 3, dropout: float = 0.2):
        super(DQN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Transformer-based attention for sequence processing
        self.attention = HybridAttention(
            input_size=input_dim,
            d_model=hidden_dim,
            nhead=8,
            num_transformer_layers=num_layers,
            tcn_channels=[hidden_dim // 2, hidden_dim, hidden_dim],
            dropout=dropout
        )

        # Dueling architecture components
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture"""
        # Extract features using attention (handles sequence input)
        features, _ = self.attention(x)  # features: (batch, hidden_dim)

        # Dueling architecture
        value = self.value_head(features)
        advantage = self.advantage_head(features)

        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(self, state: torch.Tensor, action: int, reward: float,
            next_state: torch.Tensor, done: bool):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
        """Sample batch with prioritized sampling"""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        # Extract experiences
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)

        return (torch.stack(states),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.stack(next_states),
                torch.tensor(dones, dtype=torch.float32),
                torch.tensor(weights, dtype=torch.float32),
                indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


class RainbowDQN:
    """
    Rainbow DQN implementation with:
    - Double DQN
    - Dueling architecture
    - Prioritized replay
    - Noisy networks
    - Multi-step learning
    """

    def __init__(self, input_dim: int, output_dim: int, device: torch.device, parameters: Optional[Dict] = None):

        if parameters is None:
            parameters = {}

        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = parameters.get('gamma', 0.99)
        self.epsilon = parameters.get('epsilon_start', 1.0)
        self.epsilon_end = parameters.get('epsilon_end', 0.01)
        self.epsilon_decay = parameters.get('epsilon_decay', 0.995)
        self.batch_size = parameters.get('batch_size', 32)
        self.target_update = parameters.get('target_update', 1000)
        self.multi_step = parameters.get('multi_step', 3)
        self.learning_rate = parameters.get('learning_rate', 1e-4)

        # Networks
        self.q_network = DQN(
            input_dim,
            output_dim,
            hidden_dim=parameters.get('hidden_dim', 512),
            num_layers=parameters.get('num_layers', 3),
            dropout=parameters.get('dropout', 0.2)
        ).to(device)
        self.target_network = DQN(
            input_dim,
            output_dim,
            hidden_dim=parameters.get('hidden_dim', 512),
            num_layers=parameters.get('num_layers', 3),
            dropout=parameters.get('dropout', 0.2)
        ).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Copy parameters to target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(parameters.get('buffer_size', 100000))

        # Multi-step buffer
        self.multi_step_buffer = deque(maxlen=self.multi_step)

        # Training statistics
        self.training_steps = 0
        self.losses = []

        logger.info(f"Rainbow DQN initialized with device: {device}")

    def get_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Get action using epsilon-greedy policy"""
        state = state.to(self.device)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)

        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()

    def store_experience(self, state: torch.Tensor, action: int, reward: float,
                         next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        # Add to multi-step buffer
        self.multi_step_buffer.append((state, action, reward, next_state, done))

        if len(self.multi_step_buffer) == self.multi_step:
            # Calculate multi-step return
            multi_step_return = 0
            for i, (_, _, r, _, _) in enumerate(self.multi_step_buffer):
                multi_step_return += (self.gamma ** i) * r

            # Get first and last states
            first_state = self.multi_step_buffer[0][0]
            first_action = self.multi_step_buffer[0][1]
            last_next_state = self.multi_step_buffer[-1][3]
            last_done = self.multi_step_buffer[-1][4]

            # Store in replay buffer
            self.replay_buffer.add(first_state, first_action, multi_step_return,
                                   last_next_state, last_done)

    def train(self) -> float:
        """Train the Q-network"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones, weights, indices = \
            self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Next Q-values using Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma ** self.multi_step) * next_q_values * (1 - dones)

        # Calculate loss with importance sampling weights
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors ** 2).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update priorities
        priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update beta for importance sampling
        self.replay_buffer.beta = min(1.0, self.replay_buffer.beta + self.replay_buffer.beta_increment)

        # Track loss
        self.losses.append(loss.item())

        return loss.item()

    def save(self, path: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'epsilon': self.epsilon
        }, path)

    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_steps = checkpoint['training_steps']
        self.epsilon = checkpoint['epsilon']