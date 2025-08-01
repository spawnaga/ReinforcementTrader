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
from sklearn.preprocessing import StandardScaler

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
        
        # First, check if file has headers by reading first line
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
            
        # Check if first value is a large number (nanosecond timestamp)
        first_value = first_line.split(',')[0]
        has_headers = not (first_value.isdigit() and len(first_value) > 15)
        
        if has_headers:
            # Load with headers
            df = pd.read_csv(filepath)
        else:
            # Load without headers and assign column names
            # Based on user's data format
            column_names = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + \
                         [f'feature_{i}' for i in range(10)]  # Additional columns
            df = pd.read_csv(filepath, header=None, names=column_names[:len(first_line.split(','))])
        
        # Convert timestamp if it's in nanoseconds
        if 'timestamp' in df.columns:
            # First, try to convert timestamp to numeric to check if it's nanoseconds
            try:
                # Convert to numeric for comparison
                first_timestamp = pd.to_numeric(df['timestamp'].iloc[0])
                
                # Check if timestamp is in nanoseconds (very large numbers)
                if first_timestamp > 1e15:
                    logger.info("Converting nanosecond timestamps to datetime...")
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ns')
                elif not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    # Try to convert to datetime if not already
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            except (ValueError, TypeError):
                # If conversion to numeric fails, try direct datetime conversion
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
        logger.info(f"Loaded {len(df)} rows")
        if 'timestamp' in df.columns:
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
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
        'entropy_coef': 0.1,  # Further increased to 0.1 per Grok AI recommendation
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
        'epochs_per_loop': args.epochs_per_loop,
        'max_trades_per_episode': 30,  # Further increased to 30 per Grok AI
        'min_holding_periods': 5,  # Reduced from 10 to allow quicker trades (Grok AI)
        'fill_probability': 1.0  # Set to 1.0 to ensure all orders fill (Grok AI)
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
    
    # Ensure timestamp column exists
    if 'timestamp' not in train_data.columns:
        logger.error("No timestamp column found in training data!")
        return
    
    # Rename price columns if they have database names
    if 'close_price' in train_data.columns:
        train_data = train_data.rename(columns={
            'open_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'close_price': 'close'
        })
    
    logger.info(f"Creating TimeSeriesState objects with window size {window_size}")
    
    # Limit states for initial training to avoid memory issues
    max_states = 10000  # Start with 10k states instead of 3.4M
    step_size = max(1, (len(train_data) - window_size) // max_states)
    
    logger.info(f"Creating {max_states} states with step size {step_size} (from {len(train_data)} rows)")
    
    # Initialize StandardScaler and fit on entire training data (Grok AI recommendation)
    scaler = StandardScaler()
    
    # CRITICAL FIX: Preserve actual prices for trading
    # Store actual price columns before normalization
    train_data = train_data.copy()  # Create a copy to avoid SettingWithCopyWarning
    train_data['actual_open'] = train_data['open']
    train_data['actual_high'] = train_data['high']
    train_data['actual_low'] = train_data['low']
    train_data['actual_close'] = train_data['close']
    
    # Identify numeric columns to normalize (exclude OHLC and actual price columns per Grok AI)
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'actual_open', 'actual_high', 'actual_low', 'actual_close']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Fit scaler on entire training dataset once (only non-price features)
    if len(numeric_cols) > 0:
        logger.info(f"Fitting scaler on {len(train_data)} rows of training data...")
        logger.info(f"Normalizing only technical indicators: {numeric_cols}")
        scaler.fit(train_data[numeric_cols])
        
        # Apply normalization to the entire dataset
        train_data[numeric_cols] = scaler.transform(train_data[numeric_cols])
        
        # Log normalization statistics
        logger.info(f"Normalized {len(numeric_cols)} technical indicator columns.")
        if numeric_cols:
            logger.info(
                f"Example normalized feature - {numeric_cols[0]}: mean={scaler.mean_[0]:.2f}, "
                f"std={scaler.scale_[0]:.2f}"
            )
    
    # Create states with sliding window
    states_created = 0
    for i in range(window_size, len(train_data), step_size):
        if states_created >= max_states:
            break
            
        window_data = train_data.iloc[i-window_size:i].copy()
        
        # No need to transform here - already normalized the entire dataset
        # CRITICAL: Use actual_close for price, keep normalized features for neural network
        state = TimeSeriesState(
            data=window_data,
            close_price_identifier='actual_close',  # Use actual prices for trading
            timestamp_identifier='timestamp'
        )
        states.append(state)
        states_created += 1
        
        # Log first state features for debugging (Grok AI recommendation)
        if states_created == 1 and len(numeric_cols) > 0:
            state_data = window_data[numeric_cols].values
            logger.info(
                f"Sample state features after normalization: "
                f"min={np.min(state_data):.2f}, max={np.max(state_data):.2f}, "
                f"mean={np.mean(state_data):.2f}, std={np.std(state_data):.2f}"
            )
        
        # Log first state features for debugging (Grok AI recommendation)
        if states_created == 1:
            logger.info(
                f"Sample state features after normalization: "
                f"actual_close={window_data['actual_close'].iloc[-1]:.2f}, "
                f"normalized features: {list(window_data[numeric_cols].iloc[-1].values[:3]) if numeric_cols else 'None'}"
            )
        
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
    
    # Create environment with ALL config parameters - CRITICAL FIX
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=config['value_per_tick'],
        tick_size=config['tick_size'],
        execution_cost_per_order=config['execution_cost_per_order'],
        min_holding_periods=config['min_holding_periods'],
        max_trades_per_episode=config['max_trades_per_episode'], 
        slippage_ticks=config['slippage_ticks'],
        fill_probability=config.get('fill_probability', 0.95),
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
    # Use get_loggers() instead of setup_logging() to avoid duplicate handlers
    loggers = get_loggers()
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
    
    # Track consecutive no-trade episodes (Grok AI recommendation)
    no_trade_episodes = 0
    
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
        
        # Debug: Check initial state
        if episode > 45:  # Debug episodes around 50
            loggers['algorithm'].debug(f"Episode {episode} started, curriculum stage: {env.episode_number}")
        
        # Create step progress bar
        step_pbar = tqdm(total=config['max_steps'], desc=f"Episode {episode+1}", 
                        unit="steps", leave=False,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]')
        
        # Debug: Track rewards for no-trade episodes
        step_rewards = [] if (episode > 45 and episode < 55) else None
        
        while not done and step < config['max_steps']:
            # Get action from algorithm
            action = algorithm.get_action(state)
            
            # Get current timestamp from the environment's current TimeSeriesState
            # The state returned by env is a numpy array, not a TimeSeriesState
            # We need to get the timestamp from the actual TimeSeriesState in the env
            if hasattr(env, 'states') and env.current_index < len(env.states):
                current_state_obj = env.states[env.current_index]
                timestamp = current_state_obj.ts if hasattr(current_state_obj, 'ts') else datetime.now()
                # Get price from the TimeSeriesState object
                current_price = float(current_state_obj.price) if hasattr(current_state_obj, 'price') else 0.0
            else:
                # Fallback if we can't access the states
                timestamp = datetime.now()
                current_price = float(env.current_price) if hasattr(env, 'current_price') and env.current_price is not None else 0.0
            
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
            
            # Enhanced debugging based on Grok AI's recommendations
            # 1. Log raw reward from env.step()
            if episode >= 60:
                loggers['algorithm'].debug(f"Episode {episode}, Step {step}: Raw reward from env.step() = {reward:.2f}")
                
                # 2. Check state values for price-like values
                if isinstance(state, np.ndarray):
                    state_min = np.min(state)
                    state_max = np.max(state)
                    price_like_values = state[(state >= 1000) & (state <= 5000)]
                    if len(price_like_values) > 0:
                        loggers['algorithm'].warning(
                            f"Episode {episode}, Step {step}: State contains price-like values! "
                            f"Found {len(price_like_values)} values in range [1000, 5000]: {price_like_values[:3]}"
                        )
                        loggers['algorithm'].warning(
                            f"  State range: [{state_min:.2f}, {state_max:.2f}], shape: {state.shape}"
                        )
                
                # 3. Track episode reward accumulation
                before_accumulation = episode_reward
                loggers['algorithm'].debug(
                    f"Episode {episode}, Step {step}: Before accumulation: episode_reward={before_accumulation:.2f}"
                )
            
            # Debug: Check if reward is actually a large value from env
            if episode >= 50 and abs(reward) > 100:
                loggers['algorithm'].error(
                    f"Episode {episode} LARGE RAW REWARD from env.step(): reward={reward:.2f}"
                )
                # Check for our suspicious 1214 pattern
                if 1200 <= abs(reward) <= 1250:
                    loggers['algorithm'].error(
                        f"*** FOUND PRICE-LIKE REWARD: {reward:.2f} matches NQ price range! ***"
                    )
                    loggers['algorithm'].error(
                        f"  Current price: {current_price:.2f}"
                    )
                    loggers['algorithm'].error(
                        f"  Env position: {env.current_position}"
                    )
                    loggers['algorithm'].error(
                        f"  Trades this episode: {env.trades_this_episode}"
                    )
            
            # Debug specific episode 52
            if episode == 52 and step < 10:
                loggers['algorithm'].error(
                    f"Episode 52 TRACE - Step {step}: action={action_names[action]}, "
                    f"raw_reward={reward:.2f}, position={env.current_position}, "
                    f"trades={env.trades_this_episode}, done={done}"
                )
            
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
            
            # Validate rewards for no-trade episodes (Grok AI recommendation)
            if hasattr(env, 'trades_this_episode') and env.trades_this_episode == 0 and reward != 0:
                loggers['algorithm'].warning(
                    f"Non-zero reward {reward:.2f} in no-trade episode {episode}, step {step}, "
                    f"position: {env.current_position}, last_position: {env.last_position}"
                )
                # Force zero reward for no-trade situations
                reward = 0.0
            
            # Store experience
            algorithm.store_experience(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Enhanced debugging: Log episode reward after accumulation
            if episode >= 60:
                loggers['algorithm'].debug(
                    f"Episode {episode}, Step {step}: After accumulation: episode_reward={episode_reward + reward:.2f}"
                )
                
                # Try to get algorithm's value prediction
                try:
                    if hasattr(algorithm, 'critic') and hasattr(algorithm, '_state_to_tensor'):
                        with torch.no_grad():
                            state_tensor = algorithm._state_to_tensor(state)
                            value = algorithm.critic(state_tensor).item()
                            if abs(value) > 1000:
                                loggers['algorithm'].warning(
                                    f"Episode {episode}, Step {step}: Critic value prediction = {value:.2f} "
                                    f"(PRICE-LIKE VALUE!)"
                                )
                except Exception as e:
                    pass  # Ignore errors in value prediction
            
            episode_reward += reward
            step += 1
            
            # Debug: Track rewards for suspicious episodes
            if step_rewards is not None:
                step_rewards.append(reward)
                
            # Debug: Track large step rewards and suspicious accumulation
            if abs(reward) > 100 or (episode > 45 and step < 5):
                loggers['algorithm'].debug(
                    f"Episode {episode}, Step {step}: reward={reward:.2f}, "
                    f"episode_reward={episode_reward:.2f}, trades={env.trades_this_episode}"
                )
            
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
        
        # Debug: Log final episode reward (Grok AI recommendation)
        if episode >= 60:
            loggers['algorithm'].info(
                f"Episode {episode} FINAL: episode_reward={episode_reward:.2f}, "
                f"trades={env.trades_this_episode}, steps={step}"
            )
            # Check for our suspicious 1214 pattern
            if 1200 <= abs(episode_reward) <= 1250 and env.trades_this_episode == 0:
                loggers['algorithm'].error(
                    f"*** FOUND 1214 BUG: Episode {episode} reward={episode_reward:.2f} with 0 trades! ***"
                )
        
        # Track episode completion
        if tracker:
            tracker.end_episode(episode_reward, step)
        
        all_rewards.append(episode_reward)
        
        # Track consecutive no-trade episodes and reset policy if needed (Grok AI recommendation)
        if env.trades_this_episode == 0:
            no_trade_episodes += 1
            loggers['algorithm'].warning(
                f"No trades in episode {episode}. Consecutive no-trade episodes: {no_trade_episodes}"
            )
            
            # Reset policy if too many consecutive no-trade episodes
            if no_trade_episodes >= 3:
                loggers['algorithm'].warning(
                    f"POLICY RESET: {no_trade_episodes} consecutive no-trade episodes. "
                    f"Re-initializing policy network to encourage exploration."
                )
                # Reset the policy network weights using ActorCritic's _init_weights
                algorithm.policy_network.apply(algorithm.policy_network._init_weights)
                # Reset the optimizer using adaptive learning rate (Grok AI)
                algorithm.optimizer = torch.optim.Adam(
                    algorithm.policy_network.parameters(), 
                    lr=algorithm.adaptive_lr
                )
                no_trade_episodes = 0
        else:
            no_trade_episodes = 0  # Reset counter when trades occur
        
        # Debug reward accumulation
        if abs(episode_reward) > 10000 or (episode > 45 and env.trades_this_episode == 0):
            loggers['algorithm'].warning(
                f"Episode {episode} completed with unusual reward: ${episode_reward:.2f} | "
                f"Steps: {step} | Trades: {env.trades_this_episode if hasattr(env, 'trades_this_episode') else 'N/A'} | "
                f"First reward: {all_rewards[0] if all_rewards else 'N/A'}"
            )
            
            # Log detailed step rewards for debugging
            if step_rewards is not None and env.trades_this_episode == 0:
                loggers['algorithm'].warning(
                    f"Step rewards analysis for no-trade episode {episode}:"
                )
                loggers['algorithm'].warning(
                    f"  First 10 rewards: {step_rewards[:10] if len(step_rewards) >= 10 else step_rewards}"
                )
                loggers['algorithm'].warning(
                    f"  Sum of all rewards: {sum(step_rewards):.2f}"
                )
                loggers['algorithm'].warning(
                    f"  Number of non-zero rewards: {sum(1 for r in step_rewards if r != 0)}"
                )
                
            # Check if observation contains large values
            if abs(episode_reward) > 10000:
                loggers['algorithm'].error(
                    f"Episode {episode} ANOMALY: Checking state values..."
                )
                if hasattr(state, 'shape'):
                    loggers['algorithm'].error(f"  State shape: {state.shape}")
                    loggers['algorithm'].error(f"  State min: {np.min(state):.2f}, max: {np.max(state):.2f}")
                    # Check for values close to episode_reward
                    for i, val in enumerate(state):
                        if abs(val - episode_reward) < 100:
                            loggers['algorithm'].error(
                                f"  State[{i}] = {val:.2f} is suspiciously close to episode_reward {episode_reward:.2f}!"
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