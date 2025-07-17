#!/usr/bin/env python3
"""Test if the environment is returning observation as reward"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState

# Create minimal test states
states = []
for i in range(250):
    price = 3350 + i * 0.5
    state = TimeSeriesState(
        data=[[price-5, price+5, price-3, price, 1000]],
        close_price_identifier=3,
        timestamp_identifier=None
    )
    states.append(state)

# Create environment
env = RealisticFuturesEnv(
    states=states,
    tick_size=0.25,
    value_per_tick=5.0,
    execution_cost_per_order=5.0,
    min_holding_periods=10,
    max_trades_per_episode=10
)

# Set episode number to trigger the bug
env.episode_number = 61

# Reset environment
obs = env.reset()
print(f"Initial observation shape: {obs.shape}")
print(f"Initial observation sum: {np.sum(obs):.2f}")
print(f"Initial observation first 5 values: {obs[:5]}")

# Take a few hold actions
for step in range(5):
    action = 1  # HOLD
    obs, reward, done, info = env.step(action)
    
    print(f"\nStep {step}:")
    print(f"  Observation sum: {np.sum(obs):.2f}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Trades: {env.trades_this_episode}")
    
    # Check if observation sum matches the suspicious reward values
    if 1200 < np.sum(obs) < 1600:
        print(f"  *** OBSERVATION SUM ({np.sum(obs):.2f}) IS IN SUSPICIOUS RANGE! ***")
    
    if reward > 100:
        print(f"  *** LARGE REWARD DETECTED: {reward:.2f} ***")
        
print(f"\nEnvironment episode number: {env.episode_number}")
print(f"Environment trades this episode: {env.trades_this_episode}")