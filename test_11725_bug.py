#!/usr/bin/env python3
"""
Test script to reproduce the 11,725 reward bug
"""
import numpy as np
import pandas as pd
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create realistic NQ data
print("Creating test data...")
timestamps = pd.date_range(start='2021-06-01', periods=500, freq='1min')
base_price = 14880  # NQ futures price level
prices = base_price + np.cumsum(np.random.randn(500) * 2.5)

data = pd.DataFrame({
    'timestamp': timestamps,
    'open': prices,
    'high': prices + 1,
    'low': prices - 1, 
    'close': prices,
    'volume': 100
})

# Add dummy indicators
for i in range(10):
    data[f'indicator_{i}'] = 0.0

# Create TimeSeriesState objects
states = []
window_size = 50
for i in range(window_size, len(data)):
    window_data = data.iloc[i-window_size:i].copy()
    state = TimeSeriesState(
        data=window_data,
        close_price_identifier='close',
        timestamp_identifier='timestamp'
    )
    states.append(state)

print(f"Created {len(states)} states")
print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

# Test episode 49 (EASY) vs episode 50 (MEDIUM)
for episode_num in [49, 50, 51]:
    print(f"\n{'='*60}")
    print(f"Testing Episode {episode_num}")
    print(f"{'='*60}")
    
    # Create environment
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=5.0,
        tick_size=0.25,
        execution_cost_per_order=5.0,
        min_holding_periods=5,
        max_trades_per_episode=10,
        slippage_ticks=2,
        enable_trading_logger=False
    )
    
    # Set episode number BEFORE reset
    env.episode_number = episode_num
    
    # Reset environment (this applies curriculum learning)
    initial_obs = env.reset()
    
    print(f"After reset:")
    print(f"  Episode number: {env.episode_number}")
    print(f"  Max trades: {env.max_trades_per_episode}")
    print(f"  Min holding: {env.min_holding_periods}")
    print(f"  Episode length: {env.limit}")
    print(f"  Initial observation shape: {initial_obs.shape}")
    print(f"  Initial observation values: {initial_obs[:5]}...")  # First 5 values
    
    # Check if observation contains price-like values
    if any(abs(val) > 10000 for val in initial_obs):
        print(f"  WARNING: Observation contains large values! Max: {max(initial_obs)}, Min: {min(initial_obs)}")
    
    # Take only HOLD actions (action=1) for entire episode
    total_reward = 0
    step = 0
    done = False
    
    print(f"\nRunning episode with only HOLD actions...")
    while not done and step < 10:  # Just first 10 steps
        obs, reward, done, info = env.step(1)  # HOLD action
        total_reward += reward
        
        if step < 3 or abs(reward) > 100:  # Print first 3 steps or large rewards
            print(f"  Step {step}: reward={reward:.4f}, total={total_reward:.4f}, trades={env.trades_this_episode}")
        
        step += 1
    
    # Run rest of episode without printing
    while not done:
        obs, reward, done, info = env.step(1)
        total_reward += reward
        step += 1
    
    print(f"\nEpisode {episode_num} Summary:")
    print(f"  Total steps: {step}")
    print(f"  Total trades: {env.trades_this_episode}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Expected no-trade penalty: {step * (-0.05 if episode_num < 50 else -0.075):.2f}")
    print(f"  Difference: {total_reward - (step * (-0.05 if episode_num < 50 else -0.075)):.2f}")
    
    # Check if this matches the mysterious value
    if abs(total_reward - 11725) < 100:
        print(f"  *** FOUND IT! Total reward {total_reward:.2f} is close to 11,725! ***")