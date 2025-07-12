import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import deque
import math

from .genetic_optimizer import GeneticOptimizer
from .q_learning import DQN
from .transformer_attention import TransformerAttention

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    Revolutionary Actor-Critic network with Transformer attention and multi-scale processing
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, attention_heads: int = 8,
                 attention_dim: int = 256, transformer_layers: int = 6):
        super(ActorCritic, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_dim)

        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
        ])

        # Transformer attention for market regime detection
        self.transformer_attention = TransformerAttention(
            d_model=hidden_dim,
            nhead=attention_heads,
            num_layers=transformer_layers,
            dim_feedforward=attention_dim * 4
        )

        # Market regime classifier
        self.regime_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # bull, bear, sideways, volatile
        )

        # Risk-aware feature processing
        self.risk_processor = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),  # +4 for regime features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3)  # buy, hold, sell
        )

        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        # Q-network for hybrid approach
        self.q_network = DQN(hidden_dim, 3)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize network weights with smaller values to prevent gradient explosion"""
        if isinstance(m, nn.Linear):
            # Use smaller initialization for stability
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, sequence_length: int = 60) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            sequence_length: Length of the sequence for attention

        Returns:
            action_probs: Action probabilities
            state_values: State values
            q_values: Q-values
            regime_probs: Market regime probabilities
        """
        batch_size = x.size(0)

        # Check for NaN values in input
        if torch.isnan(x).any():
            logger.warning("NaN values detected in input, replacing with zeros")
            x = torch.nan_to_num(x, nan=0.0)

        # Normalize input
        if x.dim() == 2:
            x = self.input_norm(x)
        else:
            # For 3D input, normalize along the last dimension
            orig_shape = x.shape
            x = x.view(-1, self.input_dim)
            x = self.input_norm(x)
            x = x.view(orig_shape)

        # Clamp input values to prevent extreme values
        x = torch.clamp(x, min=-10.0, max=10.0)

        # Multi-scale feature extraction
        features = []
        for extractor in self.feature_extractors:
            if x.dim() == 3:
                # Process sequence
                feat = extractor(x.view(-1, self.input_dim))
                feat = feat.view(batch_size, sequence_length, -1)
            else:
                feat = extractor(x)
            features.append(feat)

        # Combine multi-scale features
        if x.dim() == 3:
            combined_features = torch.cat(features, dim=-1)
            combined_features = combined_features.mean(dim=-1)  # Average across scales
        else:
            combined_features = torch.stack(features, dim=1).mean(dim=1)

        # Apply transformer attention for temporal patterns
        if x.dim() == 3:
            attended_features = self.transformer_attention(combined_features)
            attended_features = attended_features.mean(dim=1)  # Global average pooling
        else:
            # For single timestep, use the combined features directly
            attended_features = combined_features

        # Market regime detection
        regime_logits = self.regime_classifier(attended_features)
        regime_probs = F.softmax(regime_logits, dim=-1)

        # Risk-aware processing
        risk_features = torch.cat([attended_features, regime_probs], dim=-1)
        processed_features = self.risk_processor(risk_features)

        # Actor and Critic outputs
        action_logits = self.actor(processed_features)

        # Check for NaN in logits and clamp to prevent extreme values
        if torch.isnan(action_logits).any():
            logger.warning("NaN detected in action logits, resetting to zeros")
            action_logits = torch.zeros_like(action_logits)
        action_logits = torch.clamp(action_logits, min=-10.0, max=10.0)

        action_probs = F.softmax(action_logits, dim=-1)
        state_values = self.critic(processed_features.detach())  # Detach to prevent critic grads affecting actor
        # Q-values for hybrid approach
        q_values = self.q_network(processed_features.detach())  # Detach for Q as well

        # Final NaN check on outputs
        if torch.isnan(action_probs).any():
            logger.warning("NaN in action probs, using uniform distribution")
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)

        if torch.isnan(state_values).any():
            logger.warning("NaN in state values, resetting to zeros")
            state_values = torch.zeros_like(state_values)

        if torch.isnan(q_values).any():
            logger.warning("NaN in Q values, resetting to zeros")
            q_values = torch.zeros_like(q_values)

        return action_probs, state_values, q_values, regime_probs


class ANEPPO:
    """
    Adaptive NeuroEvolution PPO - Revolutionary hybrid algorithm combining:
    - Proximal Policy Optimization (PPO)
    - Genetic Algorithm for hyperparameter optimization
    - Q-learning for action value estimation
    - Transformer attention for market regime detection
    """

    def __init__(self, env, device: torch.device, learning_rate: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_range: float = 0.2,
                 entropy_coef: float = 0.01, value_loss_coef: float = 0.5,
                 max_grad_norm: float = 0.5, n_steps: int = 2048, batch_size: int = 64,
                 n_epochs: int = 10, genetic_population_size: int = 50,
                 genetic_mutation_rate: float = 0.1, genetic_crossover_rate: float = 0.8,
                 attention_heads: int = 8, attention_dim: float = 256, transformer_layers: int = 6):

        self.env = env
        self.device = device
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Determine input dimension from environment
        self.input_dim = self._get_input_dimension()

        # Initialize networks
        self.policy_network = ActorCritic(
            input_dim=self.input_dim,
            attention_heads=attention_heads,
            attention_dim=attention_dim,
            transformer_layers=transformer_layers
        ).to(device)

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)

        # Genetic optimizer for hyperparameter evolution
        self.genetic_optimizer = GeneticOptimizer(
            population_size=genetic_population_size,
            mutation_rate=genetic_mutation_rate,
            crossover_rate=genetic_crossover_rate
        )

        # Experience buffer
        self.experience_buffer = deque(maxlen=n_steps * 10)

        # Training statistics
        self.training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'q_losses': [],
            'genetic_scores': []
        }

        # Adaptive parameters
        self.adaptive_lr = learning_rate
        self.adaptive_clip_range = clip_range
        self.performance_window = deque(maxlen=100)

        logger.info(f"ANEPPO initialized with input_dim={self.input_dim}, device={device}")

    def _get_input_dimension(self) -> int:
        """Determine input dimension from environment"""
        try:
            # Get sample state from environment
            sample_state = self.env.reset(0)
            if hasattr(sample_state, 'data'):
                if hasattr(sample_state.data, 'shape'):
                    return sample_state.data.shape[-1]
                else:
                    return len(sample_state.data.columns) if hasattr(sample_state.data, 'columns') else 20
            else:
                return 20  # Default fallback
        except Exception as e:
            logger.warning(f"Could not determine input dimension: {e}, using default 20")
            return 20

    def get_action(self, state) -> int:
        """Get action from the policy network"""
        try:
            state_tensor = self._state_to_tensor(state)

            with torch.no_grad():
                action_probs, _, q_values, regime_probs = self.policy_network(state_tensor)

                # Hybrid action selection: combine policy and Q-values
                policy_action = Categorical(action_probs).sample()
                q_action = q_values.argmax(dim=-1)

                # Adaptive action selection based on confidence
                # Ensure action_probs is a tensor, not a tuple
                if isinstance(action_probs, tuple):
                    logger.warning("action_probs is a tuple, extracting first element")
                    action_probs = action_probs[0]
                
                policy_confidence = action_probs.max()
                q_confidence = F.softmax(q_values, dim=-1).max()

                if policy_confidence > q_confidence:
                    action = policy_action
                else:
                    action = q_action

                return action.item()

        except Exception as e:
            logger.error(f"Error getting action: {e}")
            # Add more detailed error logging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return np.random.choice(3)  # Random fallback

    def _state_to_tensor(self, state) -> torch.Tensor:
        """Convert state to tensor with robust type handling"""
        try:
            # Handle None state
            if state is None:
                return torch.zeros(1, 1, self.input_dim).to(self.device)

            # Handle memoryview objects directly
            if isinstance(state, memoryview):
                # Convert memoryview to numpy array
                try:
                    data = np.frombuffer(state, dtype=np.float32)
                except:
                    # If float32 fails, try converting as bytes first
                    data = np.frombuffer(state, dtype=np.uint8).astype(np.float32)

                # Reshape to match input dimension
                if data.size >= self.input_dim:
                    current_state = data[:self.input_dim]
                else:
                    # Pad with zeros if too small
                    current_state = np.zeros(self.input_dim, dtype=np.float32)
                    current_state[:data.size] = data
                # Return with shape (1, 1, input_dim) for single timestep
                return torch.FloatTensor(current_state).reshape(1, 1, -1).to(self.device)

            elif hasattr(state, 'data'):
                # Check if state.data is a memoryview
                if isinstance(state.data, memoryview):
                    try:
                        data = np.frombuffer(state.data, dtype=np.float32)
                    except:
                        # If float32 fails, try converting as bytes first
                        data = np.frombuffer(state.data, dtype=np.uint8).astype(np.float32)

                    # Ensure 2D shape
                    if data.ndim == 1:
                        data = data.reshape(1, -1)

                elif hasattr(state.data, 'values'):
                    # DataFrame - extract only numeric columns
                    import pandas as pd
                    df = state.data.copy()  # Create a copy to avoid modifying original

                    # First convert all columns to proper numeric types
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    # Select numeric columns
                    numeric_df = df.select_dtypes(include=[np.number])

                    # If no numeric columns, try specific columns
                    if numeric_df.empty:
                        available_cols = [col for col in numeric_cols if col in df.columns]
                        if available_cols:
                            numeric_df = df[available_cols]
                        else:
                            logger.error(f"No numeric columns found in state data")
                            return torch.zeros(1, self.input_dim).to(self.device)

                    # Fill NaN values with 0
                    numeric_df = numeric_df.fillna(0)

                    # Convert to float32 numpy array
                    data = numeric_df.values.astype(np.float32)
                elif hasattr(state.data, 'shape'):
                    # NumPy array - ensure it's numeric
                    if state.data.dtype == np.object:
                        # Try to convert object array to float
                        try:
                            data = np.array(state.data, dtype=np.float32)
                        except:
                            logger.error(f"Cannot convert object array to float32")
                            return torch.zeros(1, self.input_dim).to(self.device)
                    else:
                        data = state.data.astype(np.float32)
                else:
                    # List or other - convert to numpy array
                    try:
                        data = np.array(state.data, dtype=np.float32)
                    except:
                        logger.error(f"Cannot convert state data to numpy array")
                        return torch.zeros(1, self.input_dim).to(self.device)

                # Ensure 2D shape
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                elif data.ndim > 2:
                    data = data.reshape(data.shape[0], -1)

                # For sequential data, we need to preserve the full sequence
                # The network expects (batch_size, sequence_length, features)
                if data.shape[0] > 0:
                    # Ensure the features dimension matches input_dim
                    if data.shape[1] < self.input_dim:
                        # Pad features
                        padding = np.zeros((data.shape[0], self.input_dim - data.shape[1]), dtype=np.float32)
                        data = np.concatenate([data, padding], axis=1)
                    elif data.shape[1] > self.input_dim:
                        # Truncate features
                        data = data[:, :self.input_dim]
                    
                    # Add batch dimension: (sequence_length, features) -> (1, sequence_length, features)
                    return torch.FloatTensor(data).unsqueeze(0).to(self.device)
                else:
                    # Return zeros with proper shape: (1, 1, input_dim)
                    return torch.zeros(1, 1, self.input_dim).to(self.device)

            # Handle numpy arrays directly
            elif isinstance(state, np.ndarray):
                if state.dtype == np.object:
                    try:
                        state = state.astype(np.float32)
                    except:
                        logger.error(f"Cannot convert numpy object array to float32")
                        return torch.zeros(1, 1, self.input_dim).to(self.device)
                else:
                    state = state.astype(np.float32)

                # Handle different array shapes to ensure 3D output
                if state.ndim == 1:
                    # 1D array: (features,) -> (1, 1, features)
                    state = state.reshape(1, 1, -1)
                elif state.ndim == 2:
                    # 2D array: (sequence, features) -> (1, sequence, features)
                    state = np.expand_dims(state, 0)
                elif state.ndim == 3:
                    # Already 3D, assume it's (batch, sequence, features)
                    pass
                else:
                    logger.error(f"Unexpected state shape: {state.shape}")
                    return torch.zeros(1, 1, self.input_dim).to(self.device)

                # Ensure features dimension matches input_dim
                if state.shape[-1] < self.input_dim:
                    # Pad features
                    pad_width = [(0, 0)] * (state.ndim - 1) + [(0, self.input_dim - state.shape[-1])]
                    state = np.pad(state, pad_width, constant_values=0)
                elif state.shape[-1] > self.input_dim:
                    # Truncate features
                    state = state[..., :self.input_dim]

                return torch.FloatTensor(state).to(self.device)

            else:
                # Fallback: create dummy state
                logger.warning(f"Unknown state type: {type(state)}")
                return torch.zeros(1, 1, self.input_dim).to(self.device)

        except Exception as e:
            logger.error(f"Error converting state to tensor: {e}")
            return torch.zeros(1, 1, self.input_dim).to(self.device)

    def store_experience(self, state, action: int, reward: float, next_state, done: bool):
        """Store experience in buffer"""
        try:
            state_tensor = self._state_to_tensor(state)
            next_state_tensor = self._state_to_tensor(next_state) if next_state is not None else None

            experience = {
                'state': state_tensor.cpu(),
                'action': action,
                'reward': reward,
                'next_state': next_state_tensor.cpu() if next_state_tensor is not None else None,
                'done': done
            }

            self.experience_buffer.append(experience)

        except Exception as e:
            logger.error(f"Error storing experience: {e}")

    def train(self) -> float:
        """Train the network using experiences"""
        try:
            if len(self.experience_buffer) < self.batch_size:
                return 0.0

            # Sample batch from experience buffer
            batch = list(self.experience_buffer)[-self.batch_size:]

            # Convert to tensors
            states = torch.cat([exp['state'] for exp in batch]).to(self.device)
            actions = torch.tensor([exp['action'] for exp in batch]).to(self.device)
            rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
            dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.float32).to(
                self.device)  # Changed to float32
            # Get network outputs
            action_probs, state_values, q_values, regime_probs = self.policy_network(states)

            # Detach old values
            action_probs = action_probs.detach()
            state_values = state_values.detach()
            q_values = q_values.detach()

            # Calculate advantages using GAE
            advantages = self._calculate_advantages(rewards, state_values, dones)
            returns = advantages + state_values.squeeze()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            total_loss = 0.0

            # PPO training epochs
            for epoch in range(self.n_epochs):
                torch.autograd.set_detect_anomaly(True)
                # Get current policy outputs
                current_action_probs, current_state_values, current_q_values, _ = self.policy_network(states)

                # Policy loss (PPO clipping)
                action_log_probs = torch.log(current_action_probs.gather(1, actions.unsqueeze(1)).squeeze())
                old_action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze()).detach()

                ratio = torch.exp(action_log_probs - old_action_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.adaptive_clip_range, 1.0 + self.adaptive_clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(current_state_values.squeeze(), returns)

                # Entropy loss
                entropy_loss = -torch.mean(
                    torch.sum(current_action_probs * torch.log(current_action_probs + 1e-8), dim=1))

                # Q-learning loss
                q_targets = rewards + self.gamma * current_q_values.max(dim=1)[0] * (
                            1 - dones)  # Use 1 - dones (now float)
                q_loss = F.mse_loss(current_q_values.gather(1, actions.unsqueeze(1)).squeeze(), q_targets.detach())

                # Combined loss
                loss = (policy_loss +
                        self.value_loss_coef * value_loss +
                        self.entropy_coef * entropy_loss +
                        0.1 * q_loss)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()  # Removed retain_graph=True
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

            # Update learning rate
            self.scheduler.step()

            # Genetic optimization (periodic)
            if len(self.training_stats['episode_rewards']) % 50 == 0:
                self._genetic_optimization()

            # Adaptive parameter adjustment
            self._adaptive_parameter_adjustment()

            # Update statistics
            self.training_stats['policy_losses'].append(policy_loss.item())
            self.training_stats['value_losses'].append(value_loss.item())
            self.training_stats['entropy_losses'].append(entropy_loss.item())
            self.training_stats['q_losses'].append(q_loss.item())

            return total_loss / self.n_epochs

        except Exception as e:
            logger.error(f"Error in training: {e}")
            return 0.0

    def _calculate_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Calculate advantages using Generalized Advantage Estimation"""
        try:
            advantages = torch.zeros_like(rewards)
            gae = 0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]

                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            return advantages

        except Exception as e:
            logger.error(f"Error calculating advantages: {e}")
            return torch.zeros_like(rewards)

    def _genetic_optimization(self):
        """Perform genetic optimization of hyperparameters"""
        try:
            # Create population based on current performance
            current_performance = np.mean(self.performance_window) if self.performance_window else 0

            # Evolve hyperparameters
            evolved_params = self.genetic_optimizer.evolve({
                'learning_rate': self.adaptive_lr,
                'clip_range': self.adaptive_clip_range,
                'entropy_coef': self.entropy_coef,
                'value_loss_coef': self.value_loss_coef
            }, current_performance)

            # Update parameters
            if evolved_params:
                self.adaptive_lr = evolved_params.get('learning_rate', self.adaptive_lr)
                self.adaptive_clip_range = evolved_params.get('clip_range', self.adaptive_clip_range)
                self.entropy_coef = evolved_params.get('entropy_coef', self.entropy_coef)
                self.value_loss_coef = evolved_params.get('value_loss_coef', self.value_loss_coef)

                # Update optimizer learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.adaptive_lr

                logger.info(
                    f"Genetic optimization updated parameters: lr={self.adaptive_lr:.6f}, clip={self.adaptive_clip_range:.3f}")

        except Exception as e:
            logger.error(f"Error in genetic optimization: {e}")

    def _adaptive_parameter_adjustment(self):
        """Adaptively adjust parameters based on performance"""
        try:
            if len(self.performance_window) < 10:
                return

            recent_performance = np.mean(list(self.performance_window)[-10:])
            overall_performance = np.mean(self.performance_window)

            # Adjust clip range based on performance trend
            if recent_performance > overall_performance:
                self.adaptive_clip_range = min(0.3, self.adaptive_clip_range * 1.02)
            else:
                self.adaptive_clip_range = max(0.1, self.adaptive_clip_range * 0.98)

            # Adjust learning rate based on loss trends
            if len(self.training_stats['policy_losses']) > 100:
                recent_loss = np.mean(self.training_stats['policy_losses'][-10:])
                older_loss = np.mean(self.training_stats['policy_losses'][-100:-10])

                if recent_loss > older_loss:
                    self.adaptive_lr *= 0.95
                else:
                    self.adaptive_lr *= 1.01

                # Update optimizer
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.adaptive_lr

        except Exception as e:
            logger.error(f"Error in adaptive parameter adjustment: {e}")

    def save(self, path: str):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'policy_network': self.policy_network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'training_stats': self.training_stats,
                'hyperparameters': {
                    'learning_rate': self.adaptive_lr,
                    'clip_range': self.adaptive_clip_range,
                    'entropy_coef': self.entropy_coef,
                    'value_loss_coef': self.value_loss_coef
                }
            }

            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load(self, path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(path, map_location=self.device)

            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)

            # Load hyperparameters
            hyperparams = checkpoint.get('hyperparameters', {})
            self.adaptive_lr = hyperparams.get('learning_rate', self.adaptive_lr)
            self.adaptive_clip_range = hyperparams.get('clip_range', self.adaptive_clip_range)
            self.entropy_coef = hyperparams.get('entropy_coef', self.entropy_coef)
            self.value_loss_coef = hyperparams.get('value_loss_coef', self.value_loss_coef)

            logger.info(f"Model loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading model: {e}")