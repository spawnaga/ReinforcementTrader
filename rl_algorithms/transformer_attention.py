# transformer_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        # x should be (seq_len, batch, d_model) after transpose in transformer
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        batch_size = query.size(0)
        seq_len = query.size(1)

        # Linear transformations
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.w_o(context)

        # Residual connection and layer norm
        output = self.layer_norm(output + query)

        return output


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        # Residual connection and layer norm
        return self.layer_norm(x + residual)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        # Self-attention
        x = self.self_attention(src, src, src, src_mask)

        # Feed-forward
        x = self.feed_forward(x)

        return x


class TransformerAttention(nn.Module):
    """
    Transformer-based attention mechanism for market regime detection
    and temporal pattern recognition
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float = 0.1, max_len: int = 5000):
        super(TransformerAttention, self).__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        # Market regime detection heads
        self.regime_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.regime_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 4)  # bull, bear, sideways, volatile
        )

        # Volatility prediction head
        self.volatility_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Trend strength predictor
        self.trend_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Initialize parameters
        self.init_weights()

    def init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer

        Args:
            src: Input tensor of shape (batch_size, seq_len, d_model)
            src_mask: Optional mask tensor

        Returns:
            Enhanced features with attention
        """
        # Add positional encoding
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)

        # Pass through encoder layers
        x = src
        attention_weights = []

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        # Market regime detection
        regime_features, regime_weights = self.regime_attention(x, x, x)
        regime_probs = self.regime_classifier(regime_features.mean(dim=1))

        # Volatility prediction
        volatility = self.volatility_predictor(x.mean(dim=1))

        # Trend strength prediction
        trend_strength = self.trend_predictor(x.mean(dim=1))

        # Combine all features
        enhanced_features = x + regime_features * 0.3

        return enhanced_features, {
            'regime_probs': regime_probs,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'attention_weights': regime_weights
        }

    def get_market_regime(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get market regime probabilities"""
        with torch.no_grad():
            _, outputs = self.forward(src)
            regime_probs = F.softmax(outputs['regime_probs'], dim=-1)
            regime_prediction = regime_probs.argmax(dim=-1)

            return regime_prediction, regime_probs

    def get_attention_weights(self, src: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization"""
        with torch.no_grad():
            _, outputs = self.forward(src)
            return outputs['attention_weights']


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for time series processing"""

    def __init__(self, input_size: int, output_size: int, num_channels: list,
                 kernel_size: int = 2, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                    stride=1, dilation=dilation_size,
                                    padding=(kernel_size - 1) * dilation_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)
        self.output_projection = nn.Linear(num_channels[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        x = self.network(x)
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_channels[-1])
        x = self.output_projection(x)
        return x


class HybridAttention(nn.Module):
    """
    Hybrid attention combining Transformer and TCN for optimal
    temporal pattern recognition in financial time series
    """

    def __init__(self, input_size: int, d_model: int, nhead: int,
                 num_transformer_layers: int, tcn_channels: list,
                 dropout: float = 0.1, parameters: Optional[Dict] = None):
        if parameters is None:
            parameters = {}

        d_model = parameters.get('d_model', d_model)
        nhead = parameters.get('nhead', nhead)
        num_transformer_layers = parameters.get('num_transformer_layers', num_transformer_layers)
        dim_feedforward = parameters.get('dim_feedforward', d_model * 4)
        dropout = parameters.get('dropout', dropout)
        tcn_channels = parameters.get('tcn_channels', tcn_channels)
        max_len = parameters.get('max_len', 5000)

        super(HybridAttention, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        # Transformer branch
        self.transformer = TransformerAttention(
            d_model, nhead, num_transformer_layers,
            dim_feedforward, dropout, max_len
        )

        # TCN branch
        self.tcn = TemporalConvNet(
            input_size, d_model, tcn_channels,
            kernel_size=3, dropout=dropout
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # Output heads
        self.feature_output = nn.Linear(d_model, d_model)
        self.regime_output = nn.Linear(d_model, 4)
        self.volatility_output = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Forward pass"""
        # Project input
        x_proj = self.input_projection(x)

        # Transformer branch
        transformer_features, transformer_outputs = self.transformer(x_proj)
        transformer_pooled = transformer_features.mean(dim=1)

        # TCN branch
        tcn_features = self.tcn(x)
        tcn_pooled = tcn_features.mean(dim=1)

        # Fusion
        combined = torch.cat([transformer_pooled, tcn_pooled], dim=-1)
        fused_features = self.fusion(combined)

        # Outputs
        output_features = self.feature_output(fused_features)
        regime_probs = self.regime_output(fused_features)
        volatility = self.volatility_output(fused_features)

        outputs = {
            'regime_probs': regime_probs,
            'volatility': volatility,
            'transformer_outputs': transformer_outputs,
            'attention_weights': transformer_outputs.get('attention_weights', None)
        }

        return output_features, outputs