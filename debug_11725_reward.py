#!/usr/bin/env python3
"""Debug script to find the source of the mysterious 11,725 reward"""

import numpy as np
import pandas as pd
from gym_futures.envs import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create a simple test dataset with known values
print("Creating test data with NQ futures prices around 14,880...")
num_rows = 300  # Enough for multiple episodes

# Generate realistic NQ futures data
timestamps = pd.date_range(start='2021-06-01', periods=num_rows, freq='1min')
base_price = 14880
prices = base_price + np.cumsum(np.random.randn(num_rows) * 2.5)  # Random walk

# Create DataFrame with all expected columns
data = pd.DataFrame({
    'timestamp': timestamps,
    'open': prices + np.random.randn(num_rows) * 0.5,
    'high': prices + np.abs(np.random.randn(num_rows) * 1.0),
    'low': prices - np.abs(np.random.randn(num_rows) * 1.0),
    'close': prices,
    'volume': np.random.randint(10, 100, num_rows)
})

# Add technical indicators (simplified)
for i in range(10):  # Add 10 dummy indicators
    data[f'indicator_{i}'] = np.random.randn(num_rows)

print(f"Data shape: {data.shape}")
print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

# Test environment initialization with different episode numbers
print("\n" + "="*60)
print("Testing curriculum learning transitions...")
print("="*60)

# Test with episode numbers around 50
for episode in [0, 49, 50, 51, 52]:
    print(f"\n--- Episode {episode} ---")
    
    # Create states
    states = []
    for i in range(len(data)):
        state_data = data.iloc[i].values[1:]  # Skip timestamp
        ts = TimeSeriesState(
            ts=timestamps[i],
            price=data.iloc[i]['close'],
            high=data.iloc[i]['high'],
            low=data.iloc[i]['low'],
            volume=data.iloc[i]['volume'],
            data=np.array(state_data, dtype=np.float32)
        )
        states.append(ts)
    
    # Initialize environment
    env = RealisticFuturesEnv(
        states=states,
        episode_length=200 if episode < 51 else 150,  # Curriculum learning
        max_trades_per_episode=10 if episode < 51 else 7,
        min_holding_periods=5 if episode < 51 else 8,
        cost_per_trade=2.5 if episode < 51 else 5.0,
        slippage_ticks=1 if episode < 51 else 2,
        contract_multiplier=20,
        add_current_position_to_state=True
    )
    
    # Reset and check initial state
    env.episode_number = episode
    initial_state = env.reset()
    
    print(f"Episode length: {env.episode_length}")
    print(f"Max trades: {env.max_trades_per_episode}")
    print(f"Min holding: {env.min_holding_periods}")
    print(f"Cost per trade: ${env.cost_per_trade}")
    
    # Run a few steps without trading
    total_reward = 0
    for step in range(5):
        state, reward, done, info = env.step(1)  # Hold action
        total_reward += reward
        if step == 0:
            print(f"First step reward: {reward}")
    
    print(f"Total reward after 5 steps of holding: {total_reward}")
    
    # Check if 11,725 appears anywhere
    if abs(total_reward - 11725) < 100:
        print(f"⚠️  FOUND IT! Reward close to 11,725: {total_reward}")
    
    # Check portfolio value
    print(f"Portfolio value: {env.portfolio_value}")
    print(f"Current position: {env.current_position}")
    print(f"Initial portfolio: {env.initial_portfolio_value}")
    
    # Check if first price could be the issue
    if states:
        first_price = states[0].price
        print(f"First state price: {first_price}")
        if abs(first_price - 11725) < 1000:
            print(f"⚠️  First price ({first_price}) is close to 11,725!")

print("\n" + "="*60)
print("Checking for initialization issues...")
print("="*60)

# Check if any initial values could be 11,725
env = RealisticFuturesEnv(states=states[:100])
print(f"Initial portfolio value: {env.initial_portfolio_value}")
print(f"Contract multiplier: {env.contract_multiplier}")
print(f"First 5 prices: {[s.price for s in states[:5]]}")

# Check if it's a calculation issue
test_price = 14880
test_multiplier = 20
print(f"\nTest calculation: {test_price} / {test_multiplier} = {test_price / test_multiplier}")
print(f"Could 11,725 come from: 11,725 * 1 = {11725}")
print(f"Or from: 234,500 / 20 = {234500 / 20}")

# Check various price levels that could produce 11,725
for multiplier in [1, 2, 5, 10, 20, 50]:
    result_price = 11725 * multiplier
    print(f"11,725 * {multiplier} = {result_price}")