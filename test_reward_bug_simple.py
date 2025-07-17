#!/usr/bin/env python3
"""Simplified test to find the reward bug"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv

# Create a minimal environment without TimeSeriesState
class DummyState:
    def __init__(self, price):
        self.price = price
        self.data = np.array([[price-5, price+5, price-3, price, 1000]])

# Create states with price around 3350
states = [DummyState(3350 + i * 0.5) for i in range(250)]

# Create environment
env = RealisticFuturesEnv(
    states=states,
    tick_size=0.25,
    value_per_tick=5.0
)

# Manually set environment state to trigger the bug
env.episode_number = 61
env.trades_this_episode = 0
env.current_position = "FLAT"
env.last_position = "FLAT"  
env.entry_price = None
env.exit_price = None
env._last_closed_entry_price = None
env._last_closed_exit_price = None
env.current_index = 0

# Test get_reward directly
state = states[0]
reward = env.get_reward(state)

print(f"Episode: {env.episode_number}")
print(f"Trades: {env.trades_this_episode}")
print(f"Position: {env.current_position}")
print(f"State price: {state.price}")
print(f"Reward: {reward}")
print(f"Is reward close to 1214? {1200 < reward < 1600}")

# Check what the observation looks like
env.reset()
obs = env._get_observation(state)
print(f"\nObservation shape: {obs.shape}")
print(f"Observation sum: {np.sum(obs):.2f}")
print(f"First 5 obs values: {obs[:5] if len(obs) > 5 else obs}")
print(f"Is obs sum close to 1214? {1200 < np.sum(obs) < 1600}")