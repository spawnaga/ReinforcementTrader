#!/usr/bin/env python3
"""
Standalone training script for local execution without web interface
"""
import os
import sys
import logging
import torch
import pandas as pd
from datetime import datetime

# Import core components
from data_manager import DataManager
from rl_algorithms.ane_ppo import ANE_PPO
from gym_futures.envs.futures_env import FuturesEnv
from technical_indicators import add_all_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_local():
    """Run training directly without web interface"""
    
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
    logger.info(f"Loading data from {data_file}")
    
    # Load and prepare data
    data_manager = DataManager(data_dir="data")
    
    # Load data directly
    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} rows")
    
    # Add technical indicators if not already present
    if 'RSI_14' not in df.columns:
        logger.info("Adding technical indicators...")
        df = add_all_indicators(
            df, 
            indicators=['sin_time', 'cos_time', 'sin_weekday', 'cos_weekday', 
                       'sin_hour', 'cos_hour', 'SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR']
        )
    
    # Split data
    train_size = int(0.8 * len(df))
    train_data = df[:train_size]
    test_data = df[train_size:]
    logger.info(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # Create environment
    env = FuturesEnv(
        data=train_data,
        ticker=ticker,
        initial_balance=100000,
        tick_size=config['tick_size'],
        value_per_tick=config['value_per_tick'],
        execution_cost_per_order=config['execution_cost_per_order'],
        slippage_ticks=config['slippage_ticks'],
        min_holding_periods=config['min_holding_periods']
    )
    
    # Create algorithm
    algorithm = ANE_PPO(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
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
            logger.info(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}")
    
    # Save model
    model_path = f"models/{ticker}_ane_ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(algorithm.actor_critic.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Training completed!")
    
    # Final statistics
    final_avg = sum(all_rewards[-20:]) / 20
    logger.info(f"Final 20-episode average reward: {final_avg:.2f}")

if __name__ == "__main__":
    train_local()