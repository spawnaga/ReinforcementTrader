#!/usr/bin/env python3
"""
Direct test of reward accumulation to find the 11,725 source
"""
import numpy as np
import pandas as pd
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

# Create test data
window_size = 50
total_rows = 500

# Simple data setup
data = pd.DataFrame({
    'timestamp': pd.date_range(start='2021-06-01', periods=total_rows, freq='1min'),
    'open': 14880 + np.arange(total_rows),
    'high': 14881 + np.arange(total_rows),
    'low': 14879 + np.arange(total_rows),
    'close': 14880 + np.arange(total_rows),
    'volume': 100
})

# Add indicators
for i in range(15):  # Add more columns
    data[f'indicator_{i}'] = 0.0

print(f"Data shape: {data.shape}")

# Create states
states = []
for i in range(window_size, len(data)):
    window_data = data.iloc[i-window_size:i].copy()
    state = TimeSeriesState(
        data=window_data,
        close_price_identifier='close',
        timestamp_identifier='timestamp'
    )
    states.append(state)

print(f"Created {len(states)} states")

# Test with episode 51 parameters (Medium stage)
env = RealisticFuturesEnv(
    states=states,
    value_per_tick=5.0,        # NQ value per tick
    tick_size=0.25,
    execution_cost_per_order=5.0,
    min_holding_periods=8,     # Medium stage value
    max_trades_per_episode=7,  # Medium stage value
    slippage_ticks=2,
    session_id=1
)

# Set episode number to trigger curriculum learning
env.episode_number = 51

# Reset and check initial conditions
obs = env.reset()
print(f"\n=== Episode 51 Reset ===")
print(f"Observation shape: {obs.shape}")
print(f"Episode limit (self.limit): {env.limit}")  # Should be 200 for medium stage
print(f"Max trades: {env.max_trades_per_episode}")
print(f"Min holding: {env.min_holding_periods}")

# Run episode and track rewards
episode_reward = 0
step_rewards = []

print(f"\n=== Running Episode 51 ===")
for step in range(min(150, env.limit)):  # Run for 150 steps or episode limit
    # Always HOLD to see base reward
    obs, reward, done, info = env.step(1)  # 1 = HOLD
    
    episode_reward += reward
    step_rewards.append(reward)
    
    # Print first few steps
    if step < 5:
        print(f"Step {step}: reward={reward:.6f}, episode_total={episode_reward:.6f}")
    
    if done:
        print(f"Episode ended early at step {step}")
        break

print(f"\n=== Episode 51 Summary ===")
print(f"Total steps: {len(step_rewards)}")
print(f"Episode reward: {episode_reward:.6f}")
print(f"Average reward per step: {np.mean(step_rewards):.6f}")
print(f"Reward range: [{min(step_rewards):.6f}, {max(step_rewards):.6f}]")

# Check if this matches 11,725
if abs(episode_reward - 11725) < 1:
    print("\n*** FOUND IT! Episode reward is 11,725! ***")
elif abs(episode_reward - 117.25) < 0.1:
    print("\n*** Found scaled version: 117.25 ***")
elif abs(episode_reward - 234500) < 1:
    print("\n*** Found unscaled version: 234,500 ***")

# Calculate what would produce 11,725
print(f"\n=== Analysis ===")
print(f"11,725 / {len(step_rewards)} steps = {11725 / len(step_rewards):.2f} per step")
print(f"234,500 / 20 (contract multiplier) = {234500 / 20}")

# Check if the issue is cumulative
if len(step_rewards) > 0:
    cumulative_check = 0
    for i, r in enumerate(step_rewards):
        cumulative_check += r
        if abs(cumulative_check - 11725) < 1:
            print(f"\n*** Cumulative reward reaches 11,725 at step {i}! ***")