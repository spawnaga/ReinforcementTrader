#!/usr/bin/env python3
"""
Debug script to identify the source of the 1214.79 reward bug
Based on Grok AI's analysis
"""

import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from gym_futures.envs.futures_env import FuturesEnv
from rl_algorithms.ane_ppo import ANE_PPO
from config import Config
from data_manager import DataManager
from gpu_data_loader import GPUDataLoader
from technical_indicators import TechnicalIndicators
from gym_futures.envs.utils import TimeSeriesState

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_state_values(state):
    """Analyze state values to check for price-like values"""
    if isinstance(state, np.ndarray):
        min_val = np.min(state)
        max_val = np.max(state)
        mean_val = np.mean(state)
        
        # Check if any values are in the 1200-1300 range (typical NQ prices)
        price_like_values = state[(state >= 1000) & (state <= 5000)]
        
        return {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'shape': state.shape,
            'price_like_count': len(price_like_values),
            'price_like_values': price_like_values[:5] if len(price_like_values) > 0 else []
        }
    return None

def debug_reward_calculation():
    """Debug the reward calculation to identify the source of 1214.79 values"""
    
    logger.info("Starting reward bug debugging...")
    
    # Load configuration
    config = Config()
    
    # Initialize data manager
    data_manager = DataManager(config)
    
    # Load data
    logger.info("Loading market data...")
    try:
        # Try loading from database first
        data = data_manager.load_market_data('NQ', limit=1000)
        if data is None or data.empty:
            # Fall back to GPU data loader
            loader = GPUDataLoader()
            data_files = sorted(Path('data/processed').glob('*train*.csv'))
            if data_files:
                data = loader.load_data(str(data_files[0]))
            else:
                logger.error("No training data found")
                return
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Create states
    logger.info("Creating time series states...")
    states = []
    for i in range(60, min(200, len(data))):
        state_data = data.iloc[i-60:i]
        ts_state = TimeSeriesState(
            ts=data.iloc[i]['timestamp'] if 'timestamp' in data.columns else i,
            data=state_data,
            price=float(data.iloc[i]['close']) if 'close' in data.columns else float(data.iloc[i][4])
        )
        states.append(ts_state)
    
    # Initialize environment
    env = FuturesEnv(
        initial_cash=10000,
        execution_cost_per_order=5.0,
        value_per_tick=12.5,
        tick_size=0.25,
        min_holding_period=2,
        max_trades_per_episode=10,
        states=states,
        session_id='debug_session'
    )
    
    # Initialize algorithm
    state_dim = env._get_observation(states[0]).shape[0]
    algorithm = ANE_PPO(
        state_dim=state_dim,
        action_dim=3,
        learning_rate=1e-4,
        device='cpu'
    )
    
    logger.info(f"State dimension: {state_dim}")
    
    # Run a few episodes to reproduce the bug
    for episode in range(5):
        state = env.reset()
        done = False
        step = 0
        episode_reward = 0
        rewards_log = []
        states_log = []
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Episode {episode}")
        logger.info(f"{'='*60}")
        
        while not done and step < 100:
            # Analyze state
            state_analysis = analyze_state_values(state)
            if state_analysis:
                states_log.append(state_analysis)
                if state_analysis['price_like_count'] > 0:
                    logger.warning(f"Step {step}: State contains price-like values: {state_analysis['price_like_values']}")
            
            # Get action
            action = algorithm.get_action(state)
            
            # Log pre-step info
            logger.debug(f"Step {step}: Action={action}, Position={env.current_position}")
            
            # Step environment
            next_state, reward, done, info = env.step(action)
            
            # Log raw reward from env.step()
            logger.info(f"Step {step}: Raw reward from env.step() = {reward:.2f}")
            rewards_log.append(reward)
            
            # Check if reward looks like a price
            if abs(reward) > 1000:
                logger.error(f"LARGE REWARD DETECTED: {reward:.2f}")
                logger.error(f"  Current price: {env.current_price}")
                logger.error(f"  Position: {env.current_position}")
                logger.error(f"  Entry price: {env.entry_price}")
                logger.error(f"  Trades this episode: {env.trades_this_episode}")
            
            # Get algorithm's value prediction (if available)
            try:
                with np.errstate(all='ignore'):
                    value = algorithm.critic(algorithm._state_to_tensor(state)).item()
                    logger.debug(f"Step {step}: Critic value prediction = {value:.2f}")
                    
                    # Check if value looks like a price
                    if abs(value) > 1000:
                        logger.warning(f"LARGE VALUE PREDICTION: {value:.2f}")
            except Exception as e:
                logger.debug(f"Could not get value prediction: {e}")
            
            # Accumulate reward
            episode_reward += reward
            logger.debug(f"Step {step}: Cumulative episode_reward = {episode_reward:.2f}")
            
            # Update state
            state = next_state
            step += 1
        
        logger.info(f"\nEpisode {episode} Summary:")
        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(f"  Steps: {step}")
        logger.info(f"  Trades: {env.trades_this_episode}")
        logger.info(f"  Reward distribution: min={np.min(rewards_log):.2f}, max={np.max(rewards_log):.2f}, mean={np.mean(rewards_log):.2f}")
        
        # Analyze state values
        if states_log:
            price_like_states = sum(1 for s in states_log if s['price_like_count'] > 0)
            logger.info(f"  States with price-like values: {price_like_states}/{len(states_log)}")
            
        # Check if we've reproduced the bug
        if abs(episode_reward) > 1000 and env.trades_this_episode == 0:
            logger.error(f"\n*** BUG REPRODUCED! ***")
            logger.error(f"Episode reward = {episode_reward:.2f} with 0 trades")
            logger.error(f"This matches the pattern from the logs (rewards ~1214)")
            
            # Detailed analysis
            logger.error("\nDetailed reward breakdown:")
            non_zero_rewards = [r for r in rewards_log if r != 0]
            logger.error(f"  Non-zero rewards: {len(non_zero_rewards)}")
            logger.error(f"  First 10 rewards: {rewards_log[:10]}")
            logger.error(f"  Last 10 rewards: {rewards_log[-10:]}")
            
            # Check if rewards match state values
            if states_log:
                first_state = states_log[0]
                logger.error(f"\nFirst state analysis:")
                logger.error(f"  Shape: {first_state['shape']}")
                logger.error(f"  Range: [{first_state['min']:.2f}, {first_state['max']:.2f}]")
                if first_state['price_like_count'] > 0:
                    logger.error(f"  Contains price-like values: {first_state['price_like_values']}")
            
            break

if __name__ == "__main__":
    debug_reward_calculation()