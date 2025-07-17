#!/usr/bin/env python3
"""
Standalone training script without circular imports
"""
import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Direct imports to avoid app.py
from rl_algorithms.ane_ppo import ANEPPO
from gym_futures.envs.futures_env import FuturesEnv
from gym_futures.envs.utils import TimeSeriesState
from technical_indicators import TechnicalIndicators, add_time_based_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDataLoader:
    """Simple data loader without database dependencies"""
    
    def __init__(self):
        self.data = None
        
    def load_csv(self, filepath):
        """Load data from CSV file"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} rows")
        return df

def train_standalone():
    """Run training without web dependencies"""
    
    # Configuration
    ticker = "NQ"
    data_file = "./data/processed/NQ_train_processed.csv"
    episodes = 100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Algorithm configuration
    config = {
        'algorithm': 'ane_ppo',
        'episodes': episodes,
        'max_steps': 200,
        'batch_size': 32,
        'learning_rate': 0.0003,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'transformer_layers': 4,
        'attention_dim': 256,
        'tick_size': 0.25,
        'value_per_tick': 5.0,
        'min_holding_periods': 10,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 2
    }
    
    logger.info(f"Starting {ticker} training on {device}")
    if device == "cuda":
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    loader = SimpleDataLoader()
    df = loader.load_csv(data_file)
    
    # Add technical indicators if not already present
    if 'RSI_14' not in df.columns:
        logger.info("Adding technical indicators...")
        
        # First add time-based indicators
        df = add_time_based_indicators(df)
        
        # Then add technical indicators using TechnicalIndicators class
        ti = TechnicalIndicators(df)
        df = ti.calculate_indicators(['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR'])
    
    # Split data
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Create TimeSeriesState objects from training data
    states = []
    window_size = 50  # Look back window for each state
    
    # Ensure timestamp column exists as 'time' for TimeSeriesState
    if 'timestamp' in train_data.columns and 'time' not in train_data.columns:
        train_data['time'] = train_data['timestamp']
    
    logger.info(f"Creating TimeSeriesState objects with window size {window_size}")
    
    # Create states with sliding window
    for i in range(window_size, len(train_data)):
        window_data = train_data.iloc[i-window_size:i].copy()
        state = TimeSeriesState(
            data=window_data,
            close_price_identifier='close',
            timestamp_identifier='time'
        )
        states.append(state)
    
    logger.info(f"Created {len(states)} TimeSeriesState objects")
    
    # Create environment with correct parameters
    env = FuturesEnv(
        states=states,
        value_per_tick=config['value_per_tick'],
        tick_size=config['tick_size'],
        execution_cost_per_order=config['execution_cost_per_order']
    )
    
    # Get feature count
    state = env.reset()
    if hasattr(state, 'features'):
        state_dim = len(state.features)
    elif hasattr(state, 'shape'):
        state_dim = state.shape[0] if len(state.shape) == 1 else state.shape[-1]
    else:
        state_dim = len(state) if isinstance(state, (list, np.ndarray)) else 50  # fallback
    
    logger.info(f"State dimension: {state_dim}")
    
    # Create algorithm
    algorithm = ANEPPO(
        state_dim=state_dim,
        action_dim=3,  # Buy, Sell, Hold
        config=config,
        device=device
    )
    
    logger.info("Starting training...")
    
    # Training loop
    all_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < config['max_steps']:
            # Get action from algorithm
            action = algorithm.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            algorithm.store_experience(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            step += 1
        
        # Train algorithm
        if len(algorithm.memory) >= config['batch_size']:
            algorithm.train()
        
        all_rewards.append(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / 10
            logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Steps: {step}")
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{ticker}_ane_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    
    torch.save({
        'model_state_dict': algorithm.actor_critic.state_dict(),
        'config': config,
        'final_rewards': all_rewards[-20:],
        'state_dim': state_dim,
        'action_dim': 3
    }, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed!")
    
    # Final statistics
    final_avg = sum(all_rewards[-20:]) / 20 if len(all_rewards) >= 20 else sum(all_rewards) / len(all_rewards)
    logger.info(f"Final 20-episode average reward: {final_avg:.2f}")
    logger.info(f"Best episode reward: {max(all_rewards):.2f}")
    logger.info(f"Worst episode reward: {min(all_rewards):.2f}")

if __name__ == "__main__":
    train_standalone()