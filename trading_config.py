"""
Trading Configuration System
Comprehensive configuration for trading system with runtime options
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TradingConfig:
    """Main configuration class for the trading system"""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration from file or defaults"""
        self.config_file = config_file or "trading_config.json"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return json.load(f)
        else:
            return self._create_default_config()
            
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration"""
        return {
            # Hardware configuration
            "hardware": {
                "device": "auto",  # Options: "cpu", "gpu", "auto"
                "gpu_ids": None,   # None for all GPUs, or list like [0, 1]
                "enable_mixed_precision": True,
                "num_workers": 4
            },
            
            # Trading instrument configuration
            "instrument": {
                "ticker": "NQ",    # Futures contract symbol
                "contract_specs": "auto",  # "auto" uses futures_contracts.py
                "data_source": None,       # None uses ./{ticker}/*.csv
                "alternative_sources": [   # Fallback data sources
                    "yahoo_finance",
                    "interactive_brokers"
                ]
            },
            
            # Data configuration
            "data": {
                "timeframe": None,  # None uses entire file
                "start_date": None,
                "end_date": None,
                "train_test_split": 0.8,
                "validation_split": 0.1,
                "sequence_length": 60,
                "batch_size": 32
            },
            
            # Indicators configuration
            "indicators": {
                "time_based": [
                    "sin_time", "cos_time", "sin_weekday", "cos_weekday",
                    "hour", "minute", "day_of_week"
                ],
                "technical": [
                    "SMA", "EMA", "RSI", "MACD", "BB", "ATR", "STOCH", "ADX"
                ],
                "indicator_params": {
                    "sma_period": 20,
                    "ema_period": 20,
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_nbdev": 2,
                    "atr_period": 14,
                    "stoch_fastk": 14,
                    "stoch_slowk": 3,
                    "stoch_slowd": 3,
                    "adx_period": 14
                }
            },
            
            # Algorithm configuration
            "algorithms": {
                "primary": "ane_ppo",  # Options: "ane_ppo", "dqn", "q_learning"
                "use_transformer": True,
                "use_genetic_optimizer": True,
                "algorithm_params": {
                    "ane_ppo": {
                        "learning_rate": 3e-4,
                        "gamma": 0.99,
                        "gae_lambda": 0.95,
                        "clip_range": 0.2,
                        "value_coef": 0.5,
                        "entropy_coef": 0.01,
                        "transformer_layers": 4,
                        "attention_dim": 256
                    },
                    "genetic": {
                        "population_size": 50,
                        "generations": 100,
                        "mutation_rate": 0.1,
                        "crossover_rate": 0.8
                    }
                }
            },
            
            # Environment configuration
            "environment": {
                "type": "futures_realistic",
                "max_trades_per_episode": 5,
                "min_holding_periods": 10,
                "slippage_model": "realistic",
                "fill_probability": 0.95,
                "reward_scaling": 1.0
            },
            
            # Training configuration
            "training": {
                "episodes": 1000,
                "max_steps_per_episode": 200,
                "early_stopping_patience": 50,
                "checkpoint_frequency": 10,
                "log_frequency": 1
            },
            
            # Risk management
            "risk": {
                "max_position_size": 1,
                "stop_loss_ticks": 20,
                "take_profit_ticks": 40,
                "max_daily_loss": 1000,
                "max_drawdown": 0.2
            },
            
            # Directory structure
            "directories": {
                "base_dir": "./trading_data",
                "data_dir": "{base_dir}/{ticker}/data",
                "models_dir": "{base_dir}/{ticker}/models",
                "logs_dir": "{base_dir}/{ticker}/logs",
                "checkpoints_dir": "{base_dir}/{ticker}/checkpoints",
                "results_dir": "{base_dir}/{ticker}/results"
            }
        }
        
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated path
        
        Args:
            key_path: Dot-separated path like "hardware.device"
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
        
    def set(self, key_path: str, value: Any):
        """Set configuration value by dot-separated path"""
        keys = key_path.split('.')
        config_ref = self.config
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        config_ref[keys[-1]] = value
        
    def get_directories(self, ticker: str) -> Dict[str, Path]:
        """Get all directories for a specific ticker"""
        dirs = {}
        base_dir = self.config['directories']['base_dir']
        
        for name, template in self.config['directories'].items():
            if name != 'base_dir':
                path = template.format(base_dir=base_dir, ticker=ticker)
                dirs[name] = Path(path)
                # Create directory if it doesn't exist
                dirs[name].mkdir(parents=True, exist_ok=True)
                
        return dirs
        
    def get_device(self) -> str:
        """Get device configuration (cpu/cuda)"""
        device_config = self.config['hardware']['device']
        
        if device_config == 'auto':
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        elif device_config == 'gpu':
            return 'cuda'
        else:
            return 'cpu'
            
    def get_indicators(self) -> List[str]:
        """Get combined list of all indicators to use"""
        # If technical indicators are explicitly set (not default), use only those
        technical = self.config['indicators']['technical']
        if technical and technical != ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR']:
            # User has explicitly set indicators, use only those
            return technical
        # Otherwise combine time-based and technical
        return (self.config['indicators']['time_based'] + 
                self.config['indicators']['technical'])
                
    def update_from_args(self, args: Dict[str, Any]):
        """Update configuration from command line arguments"""
        # Map arguments to configuration paths
        arg_mapping = {
            'ticker': 'instrument.ticker',
            'device': 'hardware.device',
            'episodes': 'training.episodes',
            'algorithm': 'algorithms.primary',
            'data_source': 'instrument.data_source',
            'start_date': 'data.start_date',
            'end_date': 'data.end_date'
        }
        
        for arg, path in arg_mapping.items():
            if arg in args and args[arg] is not None:
                self.set(path, args[arg])


# Global configuration instance
_global_config = None


def get_config(config_file: Optional[str] = None) -> TradingConfig:
    """Get global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = TradingConfig(config_file)
    return _global_config


def reset_config():
    """Reset global configuration"""
    global _global_config
    _global_config = None