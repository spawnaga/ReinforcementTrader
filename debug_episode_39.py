#!/usr/bin/env python3
"""
Debug script to find why episode 39 specifically triggers the 11,725 bug
"""
import numpy as np
import pandas as pd
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create test data
window_size = 50
total_rows = 8000  # Enough for 39+ episodes

# Create data similar to NQ futures
np.random.seed(42)
base_price = 14880
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2021-06-01', periods=total_rows, freq='1min'),
    'open': base_price + np.cumsum(np.random.randn(total_rows) * 5),
    'high': 0,
    'low': 0, 
    'close': 0,
    'volume': 100
})

# Fix OHLC relationships
data['high'] = data['open'] + np.abs(np.random.randn(total_rows) * 2)
data['low'] = data['open'] - np.abs(np.random.randn(total_rows) * 2)
data['close'] = data['open'] + np.random.randn(total_rows) * 3

# Add technical indicators
for i in range(15):
    data[f'indicator_{i}'] = np.random.randn(total_rows)

print(f"Created data with shape: {data.shape}")

# Create states
states = []
for i in range(window_size, len(data), 5):  # Step by 5 for faster creation
    window_data = data.iloc[i-window_size:i].copy()
    state = TimeSeriesState(
        data=window_data,
        close_price_identifier='close',
        timestamp_identifier='timestamp'
    )
    states.append(state)

print(f"Created {len(states)} states")

# Test environment
env = RealisticFuturesEnv(
    states=states,
    value_per_tick=5.0,
    tick_size=0.25,
    execution_cost_per_order=2.5,  # Default starting value
    min_holding_periods=5,         # Default starting value
    max_trades_per_episode=10,     # Default starting value  
    slippage_ticks=1,
    session_id=1
)

# Run episodes 35-45 to catch the transition
for episode in range(35, 45):
    env.episode_number = episode
    obs = env.reset()
    
    episode_reward = 0
    trades = 0
    
    # Run the episode
    for step in range(env.limit):
        # Simple trading logic - buy/sell based on observation
        if step % 20 == 0 and trades < env.max_trades_per_episode:
            action = 0 if env.position == 0 else 2  # Buy if flat, sell if long
            trades += 1
        else:
            action = 1  # Hold
            
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
        # Check for the 11,725 value
        if abs(reward) > 1000:
            print(f"\n*** LARGE REWARD DETECTED ***")
            print(f"Episode {episode}, Step {step}")
            print(f"Reward: {reward}")
            print(f"Episode total: {episode_reward}")
            print(f"Position: {env.position}")
            print(f"Trades this episode: {env.trades_this_episode}")
            print(f"Current index: {env.current_index}")
            print(f"Observation shape: {obs.shape}")
            print(f"First 5 obs values: {obs[:5]}")
            
        if done:
            break
    
    print(f"Episode {episode}: Reward={episode_reward:.2f}, Trades={trades}")
    
    # Check if this is where it breaks
    if episode_reward > 10000:
        print(f"\n*** BUG TRIGGERED AT EPISODE {episode} ***")
        print(f"Episode settings:")
        print(f"  limit: {env.limit}")
        print(f"  max_trades: {env.max_trades_per_episode}")
        print(f"  min_holding: {env.min_holding_periods}")
        print(f"  execution_cost: {env.execution_cost_per_order}")
        break