#!/usr/bin/env python3
"""
Debug script to find exact source of 11,725 reward value
"""
import numpy as np
import pandas as pd
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create test data - similar to NQ futures prices
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

print(f"Created {len(states)} states")
print(f"Price range: ${states[0].price:.2f} - ${states[-1].price:.2f}")

# Test episode 51 (where problem occurs)
env = RealisticFuturesEnv(
    states=states,
    episode_length=150,  # Curriculum learning value
    max_trades_per_episode=7,
    min_holding_periods=8,
    cost_per_trade=5.0,
    slippage_ticks=2,
    contract_multiplier=20,  # NQ contract multiplier
    add_current_position_to_state=True
)

# Set episode number
env.episode_number = 51

# Reset environment
print("\n=== RESETTING ENVIRONMENT ===")
initial_state = env.reset()
print(f"After reset: total_reward = {env.total_reward}")
print(f"Initial observation shape: {initial_state.shape}")
print(f"Initial observation max value: {np.max(np.abs(initial_state)):.2f}")

# Check if observation contains large values
if np.max(np.abs(initial_state)) > 10000:
    print("WARNING: Initial observation contains values > 10000!")
    # Find which indices have large values
    large_indices = np.where(np.abs(initial_state) > 10000)[0]
    for idx in large_indices[:5]:  # Show first 5
        print(f"  Index {idx}: value = {initial_state[idx]:.2f}")

# Take a few steps with HOLD action
print("\n=== TAKING 5 HOLD STEPS ===")
total_episode_reward = 0
for step in range(5):
    # HOLD action (action=1)
    obs, reward, done, info = env.step(1)
    total_episode_reward += reward
    
    print(f"\nStep {step}:")
    print(f"  Action: HOLD")
    print(f"  Step reward: {reward:.4f}")
    print(f"  Total episode reward: {total_episode_reward:.4f}")
    print(f"  env.total_reward: {env.total_reward:.4f}")
    print(f"  Position: {env.current_position}")
    print(f"  Trades: {env.trades_this_episode}")
    
    # Check observation
    if np.max(np.abs(obs)) > 10000:
        print(f"  WARNING: Observation contains large value: {np.max(np.abs(obs)):.2f}")

# Check if total matches suspicious value
if abs(total_episode_reward - 11725) < 1:
    print("\n*** FOUND IT! Episode reward matches 11725! ***")
elif abs(total_episode_reward - 117.25) < 0.1:
    print("\n*** Found scaled version: 117.25 (11725 / 100) ***")
elif abs(total_episode_reward - 234500) < 1:
    print("\n*** Found unscaled version: 234,500 ***")

# Try to understand the scale
print(f"\n=== ANALYSIS ===")
print(f"Contract multiplier: {env.contract_multiplier}")
print(f"Value per tick: {env.value_per_tick}")
print(f"Tick size: {env.tick_size}")
print(f"234,500 / 20 = {234500 / 20}")
print(f"11,725 * 20 = {11725 * 20}")

# Check if any environment attributes match
print(f"\n=== CHECKING ENV ATTRIBUTES ===")
attrs_to_check = ['total_net_profit', 'current_price', 'entry_price', 'exit_price']
for attr in attrs_to_check:
    if hasattr(env, attr):
        value = getattr(env, attr)
        if value is not None:
            print(f"{attr}: {value}")
            if abs(value - 11725) < 1:
                print(f"  *** {attr} matches 11725! ***")
            elif abs(value - 234500) < 1:
                print(f"  *** {attr} matches 234500! ***")