#!/usr/bin/env python3
"""
Standalone Local Training Script
Trains models without database dependencies for local execution
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gym_futures.envs.futures_env_realistic import RealisticFuturesEnv
from rl_algorithms.ane_ppo import ANE_PPO
from rl_algorithms.dqn import DQN
from technical_indicators import TechnicalIndicators
from futures_contracts import FUTURES_SPECS
from risk_manager import RiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LocalTrainingEngine:
    """Local training engine without database dependencies"""
    
    def __init__(self, ticker='NQ', device='auto', num_gpus=1):
        self.ticker = ticker
        self.device = self._setup_device(device, num_gpus)
        self.futures_spec = FUTURES_SPECS.get(ticker, FUTURES_SPECS['NQ'])
        
        # Create directories
        self.model_dir = Path(f'models/{ticker}')
        self.log_dir = Path(f'logs/{ticker}')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"LocalTrainingEngine initialized for {ticker}")
        logger.info(f"Using device: {self.device}")
        
    def _setup_device(self, device, num_gpus):
        """Setup compute device"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'gpu':
            device = 'cuda'
            
        if device == 'cuda' and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPU(s)")
            if num_gpus and num_gpus <= gpu_count:
                # Use specified number of GPUs
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(num_gpus))
                logger.info(f"Using {num_gpus} GPU(s)")
        
        return device
        
    def load_data(self, data_path):
        """Load processed data from CSV"""
        logger.info(f"Loading data from {data_path}")
        
        try:
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")
            
            # Verify required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Log feature info
            feature_cols = [col for col in df.columns if col not in required_cols]
            logger.info(f"Available features: {len(feature_cols)}")
            
            # Check for cyclical features
            cyclical_features = ['sin_time', 'cos_time', 'sin_weekday', 'cos_weekday', 'sin_hour', 'cos_hour']
            found_cyclical = [f for f in cyclical_features if f in df.columns]
            logger.info(f"Found {len(found_cyclical)} cyclical time features: {found_cyclical}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def create_states(self, df, sequence_length=60, max_states=None):
        """Create time series states from dataframe"""
        logger.info(f"Creating states with sequence length {sequence_length}")
        
        # Drop any rows with NaN
        df = df.dropna()
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Separate price columns for special handling
        price_cols = ['open', 'high', 'low', 'close']
        feature_cols = [col for col in df.columns if col not in price_cols + ['volume']]
        
        # Scale features
        scaled_features = scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
        
        # Add original price data (will be handled by environment)
        for col in price_cols + ['volume']:
            scaled_df[col] = df[col].values
            
        # Create overlapping sequences
        states = []
        for i in range(sequence_length, len(scaled_df) - 1):
            state = scaled_df.iloc[i-sequence_length:i].values
            states.append(state)
            
            if max_states and len(states) >= max_states:
                break
                
        states = np.array(states)
        logger.info(f"Created {len(states)} states with shape {states[0].shape}")
        
        return states, list(scaled_df.columns)
        
    def train(self, data_path, algorithm='ane_ppo', episodes=100, 
              use_transformer=True, use_genetic=True, **kwargs):
        """Run training"""
        
        # Load data
        df = self.load_data(data_path)
        
        # Create states
        sequence_length = kwargs.get('sequence_length', 60)
        max_states = kwargs.get('max_states', 10000)  # Limit for memory
        states, feature_names = self.create_states(df, sequence_length, max_states)
        
        # Create environment
        env = RealisticFuturesEnv(
            states=states,
            value_per_tick=self.futures_spec['value_per_tick'],
            tick_size=self.futures_spec['tick_size'],
            execution_cost_per_order=kwargs.get('execution_cost', 5.0),
            min_holding_periods=kwargs.get('min_holding', 10),
            max_trades_per_episode=kwargs.get('max_trades', 5),
            slippage_ticks=kwargs.get('slippage', 2),
            session_id=None  # No database session
        )
        
        # Get dimensions
        state_shape = env.observation_space.shape
        action_dim = env.action_space.n
        
        logger.info(f"Environment created: state_shape={state_shape}, actions={action_dim}")
        
        # Create algorithm
        if algorithm == 'ane_ppo':
            model = ANE_PPO(
                state_shape=state_shape,
                action_dim=action_dim,
                device=self.device,
                use_transformer=use_transformer,
                transformer_layers=kwargs.get('transformer_layers', 2),
                attention_dim=kwargs.get('attention_dim', 256),
                learning_rate=kwargs.get('learning_rate', 3e-4)
            )
            logger.info("Created ANE-PPO model with transformer attention")
        elif algorithm == 'dqn':
            model = DQN(
                state_shape=state_shape,
                action_dim=action_dim,
                device=self.device,
                learning_rate=kwargs.get('learning_rate', 1e-3),
                buffer_size=kwargs.get('buffer_size', 10000)
            )
            logger.info("Created DQN model")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        # Training loop
        logger.info(f"Starting training for {episodes} episodes...")
        
        all_rewards = []
        all_profits = []
        
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_profit = 0
            done = False
            step = 0
            
            while not done:
                # Get action from model
                if hasattr(model, 'get_action'):
                    action = model.get_action(state)
                else:
                    action = model.act(state)
                    
                # Take step
                next_state, reward, done, info = env.step(action)
                
                # Store experience and train
                if algorithm == 'ane_ppo':
                    model.store_transition(state, action, reward, next_state, done)
                elif algorithm == 'dqn':
                    model.remember(state, action, reward, next_state, done)
                    if len(model.memory) > model.batch_size:
                        model.replay()
                
                episode_reward += reward
                episode_profit += info.get('profit', 0)
                state = next_state
                step += 1
                
            # Train PPO at end of episode
            if algorithm == 'ane_ppo' and len(model.memory['states']) > 0:
                model.train()
                
            all_rewards.append(episode_reward)
            all_profits.append(episode_profit)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(all_rewards[-10:])
                avg_profit = np.mean(all_profits[-10:])
                logger.info(f"Episode {episode+1}/{episodes} - "
                          f"Avg Reward: {avg_reward:.2f}, "
                          f"Avg Profit: ${avg_profit:.2f}")
                          
            # Save checkpoint
            if (episode + 1) % 100 == 0:
                checkpoint_path = self.model_dir / f"{algorithm}_ep{episode+1}.pt"
                torch.save({
                    'episode': episode + 1,
                    'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
                    'avg_reward': np.mean(all_rewards[-100:]),
                    'avg_profit': np.mean(all_profits[-100:])
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
                
        # Final save
        final_path = self.model_dir / f"{algorithm}_final.pt"
        torch.save({
            'episodes': episodes,
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
            'final_avg_reward': np.mean(all_rewards),
            'final_avg_profit': np.mean(all_profits),
            'all_rewards': all_rewards,
            'all_profits': all_profits
        }, final_path)
        
        logger.info(f"\nTraining Complete!")
        logger.info(f"Final Average Reward: {np.mean(all_rewards):.2f}")
        logger.info(f"Final Average Profit: ${np.mean(all_profits):.2f}")
        logger.info(f"Model saved to: {final_path}")
        
        return model, all_rewards, all_profits


def main():
    parser = argparse.ArgumentParser(description='Local Training Script')
    
    # Basic options
    parser.add_argument('--ticker', type=str, default='NQ', help='Futures ticker')
    parser.add_argument('--data-file', type=str, required=True, help='Path to processed data CSV')
    parser.add_argument('--algorithm', choices=['ane_ppo', 'dqn'], default='ane_ppo')
    parser.add_argument('--episodes', type=int, default=100)
    
    # Hardware options
    parser.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto')
    parser.add_argument('--num-gpus', type=int, default=1)
    
    # Algorithm options
    parser.add_argument('--use-transformer', action='store_true', default=True)
    parser.add_argument('--use-genetic', action='store_true', default=True)
    
    # Training parameters
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--max-states', type=int, default=10000)
    
    # Environment parameters
    parser.add_argument('--max-trades', type=int, default=5)
    parser.add_argument('--min-holding', type=int, default=10)
    parser.add_argument('--slippage', type=int, default=2)
    
    args = parser.parse_args()
    
    # Create engine
    engine = LocalTrainingEngine(
        ticker=args.ticker,
        device=args.device,
        num_gpus=args.num_gpus
    )
    
    # Run training
    engine.train(
        data_path=args.data_file,
        algorithm=args.algorithm,
        episodes=args.episodes,
        use_transformer=args.use_transformer,
        use_genetic=args.use_genetic,
        sequence_length=args.sequence_length,
        learning_rate=args.learning_rate,
        max_states=args.max_states,
        max_trades=args.max_trades,
        min_holding=args.min_holding,
        slippage=args.slippage
    )


if __name__ == '__main__':
    main()