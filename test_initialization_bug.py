#!/usr/bin/env python3
"""
Test if 11,725 comes from initialization or first step
"""
import numpy as np
import pandas as pd
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create minimal test data
timestamps = pd.date_range(start='2021-06-01', periods=300, freq='1min')
base_price = 14880  # NQ price
prices = base_price + np.cumsum(np.random.randn(300) * 2.5)

data = pd.DataFrame({
    'timestamp': timestamps,
    'open': prices,
    'high': prices + 1,
    'low': prices - 1, 
    'close': prices,
    'volume': 100
})

# Add indicators
for i in range(10):
    data[f'indicator_{i}'] = 0.0

# Create states
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
print(f"Price at index 0: ${states[0].price:.2f}")

# Test episode 51 specifically
env = RealisticFuturesEnv(
    states=states,
    value_per_tick=5.0,
    tick_size=0.25,
    execution_cost_per_order=5.0,
    min_holding_periods=8,
    max_trades_per_episode=7,
    slippage_ticks=2,
    enable_trading_logger=False
)

# Set to episode 50 so reset() will increment to 51
env.episode_number = 50

print("\n" + "="*60)
print("Testing Episode 51 Initialization")
print("="*60)

# Check total_reward BEFORE reset
print(f"\nBEFORE reset:")
print(f"  env.total_reward = {env.total_reward}")
print(f"  env.episode_number = {env.episode_number}")

# Reset environment
initial_obs = env.reset()

print(f"\nAFTER reset:")
print(f"  env.total_reward = {env.total_reward}")
print(f"  env.episode_number = {env.episode_number}")
print(f"  env.current_index = {env.current_index}")
print(f"  Episode length = {env.limit}")

# Check if observation contains suspicious values
obs_max = np.max(initial_obs)
obs_min = np.min(initial_obs)
print(f"\nInitial observation stats:")
print(f"  Shape: {initial_obs.shape}")
print(f"  Max value: {obs_max:.2f}")
print(f"  Min value: {obs_min:.2f}")

# Look for 11,725 in observation
for i, val in enumerate(initial_obs):
    if abs(val - 11725) < 100:
        print(f"  *** FOUND 11,725 at index {i}: {val:.2f} ***")
    elif abs(val) > 10000:
        print(f"  Large value at index {i}: {val:.2f}")

# Take a single HOLD step
print(f"\nTaking first HOLD step...")
obs, reward, done, info = env.step(1)

print(f"\nAfter first step:")
print(f"  Step reward = {reward:.4f}")
print(f"  Total reward = {env.total_reward:.4f}")
print(f"  Current index = {env.current_index}")

# Check if the reward is suspicious
if abs(reward) > 1000:
    print(f"\n*** SUSPICIOUS FIRST STEP REWARD: {reward:.2f} ***")
    
# Take a few more steps
print(f"\nTaking 5 more HOLD steps...")
for i in range(5):
    obs, reward, done, info = env.step(1)
    print(f"  Step {i+2}: reward={reward:.4f}, total={env.total_reward:.4f}")
    
    if abs(reward) > 1000:
        print(f"    *** LARGE REWARD DETECTED: {reward:.2f} ***")

# Calculate what 11,725 could be
print(f"\n" + "="*60)
print("Possible sources of 11,725:")
print("="*60)
print(f"  234,500 / 20 = {234500 / 20:.0f}")
print(f"  469,000 / 40 = {469000 / 40:.0f}")
print(f"  58,625 / 5 = {58625 / 5:.0f}")
print(f"  2,930,000 / 250 = {2930000 / 250:.0f}")
print(f"  14,880 * 0.787 = {14880 * 0.787:.0f}")  # Some percentage of price?