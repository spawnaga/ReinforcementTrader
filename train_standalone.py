#!/usr/bin/env python3
"""
Standalone training script with professional logging and tracking
"""
import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import psutil
import time
from tqdm import tqdm
import uuid
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

# Direct imports to avoid app.py
from rl_algorithms.ane_ppo import ANEPPO
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState
from technical_indicators import TechnicalIndicators, add_time_based_indicators

# Import professional logging and tracking
from logging_config import setup_logging, get_loggers
from training_tracker import TrainingTracker

# Configure minimal console logging
logging.basicConfig(level=logging.WARNING)
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

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Multi-GPU Reinforcement Learning Trading')
    
    # Data arguments
    parser.add_argument('--ticker', type=str, default='NQ', help='Ticker symbol to trade')
    parser.add_argument('--data-file', type=str, default='./data/processed/NQ_train_processed.csv',
                        help='Path to training data CSV file')
    
    # Training arguments
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    
    # GPU arguments
    parser.add_argument('--num-gpus', type=int, default=None, 
                        help='Number of GPUs to use (default: use all available)')
    parser.add_argument('--gpu-ids', type=str, default=None,
                        help='Specific GPU IDs to use (e.g., "0,2,3")')
    
    # Algorithm arguments
    parser.add_argument('--algorithm', type=str, default='ANE-PPO', 
                        choices=['ANE-PPO', 'DQN'], help='Algorithm to use')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--transformer-layers', type=int, default=4, help='Transformer layers')
    parser.add_argument('--attention-dim', type=int, default=256, help='Attention dimension')
    
    # Advanced arguments
    parser.add_argument('--training-loops', type=int, default=1, help='Number of training loops')
    parser.add_argument('--epochs-per-loop', type=int, default=10, help='Epochs per training loop')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                        help='Gradient accumulation steps')
    parser.add_argument('--num-workers', type=int, default=4, help='Data loader workers')
    
    return parser.parse_args()

def setup_gpu_devices(args):
    """Setup GPU devices based on arguments"""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return "cpu", []
    
    # Get available GPUs
    available_gpus = list(range(torch.cuda.device_count()))
    logger.info(f"Available GPUs: {available_gpus}")
    
    # Determine which GPUs to use
    if args.gpu_ids:
        # Use specific GPU IDs
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
        # Validate GPU IDs
        gpu_ids = [g for g in gpu_ids if g in available_gpus]
    elif args.num_gpus:
        # Use specified number of GPUs
        gpu_ids = available_gpus[:args.num_gpus]
    else:
        # Use all available GPUs
        gpu_ids = available_gpus
    
    if not gpu_ids:
        logger.warning("No valid GPUs specified, using CPU")
        return "cpu", []
    
    # Set CUDA visible devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    
    # Log GPU configuration
    logger.info(f"Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    for i, gpu_id in enumerate(gpu_ids):
        name = torch.cuda.get_device_name(gpu_id)
        mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        logger.info(f"  GPU {gpu_id}: {name} ({mem:.1f} GB)")
    
    # Return primary device and list of devices
    device = f"cuda:{gpu_ids[0]}" if len(gpu_ids) == 1 else "cuda"
    return device, gpu_ids

def train_standalone():
    """Run training without web dependencies"""
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup GPU devices
    device, gpu_ids = setup_gpu_devices(args)
    
    # Configuration from arguments
    ticker = args.ticker
    data_file = args.data_file
    episodes = args.episodes
    
    # Algorithm configuration
    config = {
        'algorithm': args.algorithm.lower().replace('-', '_'),
        'episodes': episodes,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'transformer_layers': args.transformer_layers,
        'attention_dim': args.attention_dim,
        'tick_size': 0.25,
        'value_per_tick': 5.0,
        'min_holding_periods': 10,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 2,
        'num_gpus': len(gpu_ids),
        'gpu_ids': gpu_ids,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'training_loops': args.training_loops,
        'epochs_per_loop': args.epochs_per_loop
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
        train_data = train_data.copy()  # Create explicit copy to avoid warning
        train_data['time'] = train_data['timestamp']
    
    logger.info(f"Creating TimeSeriesState objects with window size {window_size}")
    
    # Limit states for initial training to avoid memory issues
    max_states = 10000  # Start with 10k states instead of 3.4M
    step_size = max(1, (len(train_data) - window_size) // max_states)
    
    logger.info(f"Creating {max_states} states with step size {step_size} (from {len(train_data)} rows)")
    
    # Create states with sliding window
    states_created = 0
    for i in range(window_size, len(train_data), step_size):
        if states_created >= max_states:
            break
            
        window_data = train_data.iloc[i-window_size:i].copy()
        state = TimeSeriesState(
            data=window_data,
            close_price_identifier='close',
            timestamp_identifier='time'
        )
        states.append(state)
        states_created += 1
        
        # Progress logging
        if states_created % 1000 == 0:
            logger.info(f"Created {states_created}/{max_states} states...")
            # Log memory usage
            import psutil
            process = psutil.Process()
            mem_info = process.memory_info()
            logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
    
    logger.info(f"Created {len(states)} TimeSeriesState objects")
    
    # Generate session ID early for use in environment
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + str(uuid.uuid4())[:8]
    
    # Create environment with session_id and disable old trading logger
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=config['value_per_tick'],
        tick_size=config['tick_size'],
        execution_cost_per_order=config['execution_cost_per_order'],
        session_id=session_id,
        enable_trading_logger=False  # Use our new logging system instead
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
        env=env,
        device=device,
        learning_rate=config.get('learning_rate', 0.0003),
        gamma=config.get('gamma', 0.99),
        gae_lambda=config.get('gae_lambda', 0.95),
        clip_range=config.get('clip_range', 0.2),
        entropy_coef=config.get('entropy_coef', 0.01),
        value_loss_coef=config.get('value_coef', 0.5),
        batch_size=config.get('batch_size', 32),
        n_epochs=config.get('update_epochs', 10),
        transformer_layers=config.get('transformer_layers', 4),
        attention_dim=int(config.get('attention_dim', 256))  # Convert to int
    )
    
    # Enable multi-GPU if specified
    if len(gpu_ids) > 1:
        algorithm.enable_multi_gpu(gpu_ids)
        logger.info(f"Multi-GPU training enabled with {len(gpu_ids)} GPUs")
    
    # Initialize professional logging and tracking
    loggers = setup_logging()
    # session_id already created above for use in FuturesEnv
    
    # Initialize PostgreSQL tracker
    try:
        tracker = TrainingTracker(
            session_id=session_id,
            algorithm_name='ANE-PPO',
            ticker=ticker,
            hyperparameters=config
        )
    except Exception as e:
        print(f"Note: PostgreSQL tracking not available: {e}")
        tracker = None
    
    # Log training start
    loggers['algorithm'].info("="*60)
    loggers['algorithm'].info(f"Starting ANE-PPO training for {ticker}")
    loggers['algorithm'].info(f"Episodes: {episodes}, Device: {device}")
    loggers['algorithm'].info("="*60)
    
    # Training loop with tqdm progress bar
    all_rewards = []
    
    # Create main progress bar for episodes
    episode_pbar = tqdm(range(episodes), desc="Training Episodes", unit="ep", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for episode in episode_pbar:
        if tracker:
            tracker.start_episode(episode)
            
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Create step progress bar
        step_pbar = tqdm(total=config['max_steps'], desc=f"Episode {episode+1}", 
                        unit="steps", leave=False,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]')
        
        while not done and step < config['max_steps']:
            # Get action from algorithm
            action = algorithm.get_action(state)
            
            # Get current timestamp and price from state
            timestamp = state.ts if hasattr(state, 'ts') else datetime.now()
            # Fix: Get price from state.price instead of state.current_price
            current_price = state.price if hasattr(state, 'price') else env.current_price if hasattr(env, 'current_price') else 0
            
            # Log agent's decision BEFORE taking action
            action_names = {0: 'BUY', 1: 'HOLD', 2: 'SELL'}
            position_names = {-1: 'SHORT', 0: 'FLAT', 1: 'LONG'}
            current_position_name = position_names.get(env.current_position, 'UNKNOWN')
            
            # Log to algorithm.log - what the agent decided
            loggers['algorithm'].info(
                f"Step {step} | Time: {timestamp} | Price: ${current_price:.2f} | "
                f"Position: {current_position_name} | Action: {action_names[action]} | "
                f"Episode P/L: ${episode_reward:.2f}"
            )
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Debug large rewards
            if abs(reward) > 1000:
                loggers['algorithm'].warning(
                    f"LARGE REWARD DETECTED: ${reward:.2f} at step {step}, "
                    f"position: {env.current_position}, action: {action_names[action]}"
                )
            
            # Log position changes to positions.log
            if env.current_position != env.last_position:
                # Position changed
                if env.current_position == 0:
                    # Closed a position
                    position_type = 'LONG' if env.last_position == 1 else 'SHORT'
                    entry_price = env._last_closed_entry_price
                    exit_price = env._last_closed_exit_price
                    entry_time = env.entry_time if hasattr(env, 'entry_time') else None
                    
                    if entry_price and exit_price:
                        profit = (exit_price - entry_price) * config['value_per_tick'] if position_type == 'LONG' else (entry_price - exit_price) * config['value_per_tick']
                        profit -= 2 * config['execution_cost_per_order']
                        hold_duration = step - (env._entry_step if hasattr(env, '_entry_step') else 0)
                        
                        # Log to trading.log with timestamps
                        loggers['trading'].info(
                            f"CLOSED {position_type} | Entry: {entry_time} @ ${entry_price:.2f} | "
                            f"Exit: {timestamp} @ ${exit_price:.2f} | Held: {hold_duration} steps | "
                            f"Net P/L: ${profit:.2f}"
                        )
                        
                        # Log to positions.log
                        loggers['positions'].info(
                            f"Position CLOSED | {position_type} | Duration: {hold_duration} steps | "
                            f"Entry: ${entry_price:.2f} Exit: ${exit_price:.2f} | P/L: ${profit:.2f}"
                        )
                        
                        # Log to rewards.log with time context
                        loggers['rewards'].info(
                            f"Trade Reward: ${profit:.2f} | Time: {timestamp} | "
                            f"Total Episode P/L: ${episode_reward + reward:.2f}"
                        )
                        
                        # Track trade in PostgreSQL if available
                        if tracker:
                            tracker.log_trade(
                                entry_time=entry_time or timestamp,
                                exit_time=timestamp,
                                position_type=position_type,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                profit=profit,
                                commission=config['execution_cost_per_order'],
                                slippage=0
                            )
                else:
                    # Opened a new position
                    new_position = 'LONG' if env.current_position == 1 else 'SHORT'
                    env.entry_time = timestamp  # Store entry time
                    env._entry_step = step  # Store entry step
                    
                    # Get the actual entry price with slippage from the environment
                    actual_entry_price = env.entry_price if hasattr(env, 'entry_price') and env.entry_price else current_price
                    
                    loggers['positions'].info(
                        f"Position OPENED | {new_position} | Time: {timestamp} | "
                        f"Entry Price: ${actual_entry_price:.2f}"
                    )
                    loggers['trading'].info(
                        f"OPENED {new_position} | Time: {timestamp} | Entry: ${actual_entry_price:.2f}"
                    )
            
            # Log step reward
            if reward != 0:
                loggers['rewards'].info(
                    f"Step Reward: ${reward:.2f} | Time: {timestamp} | Action: {action_names[action]}"
                )
            if tracker and step % 100 == 0:  # Log every 100 steps
                tracker.log_algorithm_decision(
                    step=step,
                    action='BUY' if action == 0 else 'SELL' if action == 2 else 'HOLD',
                    action_probs={'buy': 0.33, 'sell': 0.33, 'hold': 0.34},  # Placeholder
                    reward=reward
                )
            
            # Store experience
            algorithm.store_experience(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            step += 1
            
            # Debug check for massive rewards
            if abs(episode_reward) > 100000 and env.trades_this_episode == 0:
                loggers['algorithm'].error(
                    f"MASSIVE EPISODE REWARD DETECTED: ${episode_reward:.2f} | "
                    f"Step: {step} | Last step reward: ${reward:.2f} | "
                    f"Trades: {env.trades_this_episode} | Done: {done}"
                )
                # Check if it matches our suspicious value
                if abs(episode_reward - 117701.50) < 0.01:
                    loggers['algorithm'].error(
                        "*** FOUND THE BUG: Episode reward matches 117701.50! ***"
                    )
            
            # Update step progress bar
            step_pbar.update(1)
            step_pbar.set_postfix({'reward': f'{episode_reward:.2f}'})
        
        step_pbar.close()
        
        # Train algorithm
        if len(algorithm.experience_buffer) >= config['batch_size']:
            algorithm.train()
        
        # Track episode completion
        if tracker:
            tracker.end_episode(episode_reward, step)
        
        all_rewards.append(episode_reward)
        
        # Debug reward accumulation
        if abs(episode_reward) > 10000:
            loggers['algorithm'].error(
                f"ABNORMAL EPISODE REWARD: ${episode_reward:.2f} | "
                f"Steps: {step} | Trades: {env.trades_this_episode if hasattr(env, 'trades_this_episode') else 'N/A'}"
            )
        
        # Update episode progress bar
        avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards))
        episode_pbar.set_postfix({
            'avg_reward': f'{avg_reward:.2f}',
            'last_reward': f'{episode_reward:.2f}'
        })
        
        # Check if agent is learning (every 50 episodes)
        if tracker and (episode + 1) % 50 == 0:
            assessment = tracker.get_learning_assessment()
            if assessment.get('is_learning'):
                loggers['performance'].info(
                    f"✓ AGENT IS LEARNING! Reward improved by {assessment['reward_improvement']:.1%}"
                )
            else:
                loggers['performance'].info(
                    f"⚠ Agent not improving yet. Episode {episode+1}"
                )
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{ticker}_ane_ppo_{session_id}.pt"
    
    torch.save({
        'model_state_dict': algorithm.policy_network.state_dict(),
        'config': config,
        'final_rewards': all_rewards[-20:],
        'state_dim': algorithm.input_dim,
        'action_dim': 3
    }, model_path)
    
    # Calculate final statistics
    final_avg = sum(all_rewards[-20:]) / 20 if len(all_rewards) >= 20 else sum(all_rewards) / len(all_rewards)
    best_reward = max(all_rewards)
    worst_reward = min(all_rewards)
    
    # Save checkpoint in tracker
    if tracker:
        tracker.save_checkpoint(
            model_path=str(model_path),
            episode=episodes,
            avg_reward=final_avg,
            best_reward=best_reward,
            is_best=True
        )
    
    # Log final summary
    loggers['performance'].info("="*60)
    loggers['performance'].info("TRAINING COMPLETED!")
    loggers['performance'].info(f"Model saved to: {model_path}")
    loggers['performance'].info(f"Final 20-episode average reward: {final_avg:.2f}")
    loggers['performance'].info(f"Best episode reward: {best_reward:.2f}")
    loggers['performance'].info(f"Worst episode reward: {worst_reward:.2f}")
    
    # Get final learning assessment
    if tracker:
        final_assessment = tracker.get_learning_assessment()
        if final_assessment.get('is_learning'):
            loggers['performance'].info(f"✓ AGENT LEARNED! Reward improved by {final_assessment['reward_improvement']:.1%}")
            loggers['performance'].info(f"Win rate improved by {final_assessment['win_rate_improvement']:.1%}")
        else:
            loggers['performance'].info("⚠ Agent needs more episodes to show clear learning")
    
    loggers['performance'].info("="*60)
    
    # Close tracker
    if tracker:
        tracker.close()
        
    print("\n✓ Training completed! Check the logs/ folder for detailed logs:")
    print(f"  • logs/latest/ → symlink to current session")
    print(f"  • logs/{loggers['session_timestamp']}/ → this session's logs")
    print(f"    - trading.log    → all trade executions")
    print(f"    - positions.log  → position tracking") 
    print(f"    - rewards.log    → reward calculations")
    print(f"    - algorithm.log  → algorithm decisions")
    print(f"    - performance.log → learning metrics")

if __name__ == "__main__":
    train_standalone()