#!/usr/bin/env python3
"""
Check if 11,725 is in the observation array, not the reward
"""
import numpy as np
import pandas as pd
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create minimal test data
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2021-06-01', periods=200, freq='1min'),
    'open': [14880 + i for i in range(200)],
    'high': [14881 + i for i in range(200)],
    'low': [14879 + i for i in range(200)],
    'close': [14880 + i for i in range(200)],
    'volume': [100] * 200
})

# Add indicators
for i in range(10):
    data[f'indicator_{i}'] = 0.0

# Create states
states = []
for i in range(len(data)):
    state = TimeSeriesState(
        data=data.iloc[i:i+1],
        close_price_identifier='close',
        timestamp_identifier='timestamp'
    )
    states.append(state)

print(f"Created {len(states)} states")
print(f"First price: ${states[0].price:.2f}")
print(f"Last price: ${states[-1].price:.2f}")

# Create environment with curriculum learning parameters for episode 51
env = RealisticFuturesEnv(
    states=states,
    episode_length=150,
    max_trades_per_episode=7,
    min_holding_periods=8,
    cost_per_trade=5.0,
    slippage_ticks=2,
    contract_multiplier=20,
    add_current_position_to_state=True
)

# Set episode number and reset
env.episode_number = 51
obs = env.reset()

print(f"\n=== OBSERVATION ANALYSIS ===")
print(f"Observation shape: {obs.shape}")
print(f"Observation min: {np.min(obs):.2f}")
print(f"Observation max: {np.max(obs):.2f}")
print(f"Observation mean: {np.mean(obs):.2f}")

# Check for suspicious values
large_values = obs[np.abs(obs) > 1000]
if len(large_values) > 0:
    print(f"\n=== LARGE VALUES FOUND ===")
    print(f"Number of values > 1000: {len(large_values)}")
    print(f"Large values: {large_values[:10]}")  # Show first 10
    
    # Check if any match our suspicious values
    for val in large_values:
        if abs(val - 11725) < 1:
            print(f"*** FOUND 11,725 in observation! Value: {val} ***")
        elif abs(val - 234500) < 1:
            print(f"*** FOUND 234,500 in observation! Value: {val} ***")
        elif abs(val - 14880) < 10:
            print(f"*** Found price data ~14,880 in observation! Value: {val} ***")

# Take a step and check reward vs observation
print(f"\n=== STEP TEST ===")
obs, reward, done, info = env.step(1)  # HOLD action

print(f"Step reward: {reward:.6f}")
print(f"Observation contains large values: {np.any(np.abs(obs) > 1000)}")

if np.any(np.abs(obs) > 1000):
    print(f"Max observation value: {np.max(np.abs(obs)):.2f}")
    
# Check what the agent would see as "episode reward" if it accumulated observation values
print(f"\n=== ACCUMULATION TEST ===")
fake_accumulation = 0
for i in range(5):
    obs, reward, done, info = env.step(1)
    fake_accumulation += np.max(obs)  # If bug accumulated max observation value
    print(f"Step {i}: reward={reward:.4f}, max_obs={np.max(obs):.2f}, fake_accum={fake_accumulation:.2f}")

if abs(fake_accumulation - 11725 * 5) < 100:
    print("\n*** BUG HYPOTHESIS: Agent might be accumulating observation values! ***")