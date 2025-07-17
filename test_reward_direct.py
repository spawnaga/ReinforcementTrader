#!/usr/bin/env python3
"""Direct test of reward calculation to find the bug"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv

# Create a minimal environment 
env = RealisticFuturesEnv(
    states=[],  # We'll set this up manually
    tick_size=0.25,
    value_per_tick=5.0
)

# Disable trading logger to avoid attribute errors
env.trading_logger = None

# Manually set environment state to match bug conditions
env.episode_number = 61
env.trades_this_episode = 0
env.current_position = "FLAT"
env.last_position = "FLAT"  
env.entry_price = None
env.exit_price = None
env._last_closed_entry_price = None
env._last_closed_exit_price = None
env.current_index = 10  # Not at the beginning
env.last_trade_step = -10  # No recent trades
env.position_history = ["FLAT"] * 10  # Always been flat

# Create a dummy state with price around 3350
class DummyState:
    def __init__(self, price):
        self.price = price

state = DummyState(3350)

# Test get_reward directly
print("Testing reward calculation...")
print(f"Episode: {env.episode_number}")
print(f"Trades: {env.trades_this_episode}")
print(f"Position: {env.current_position}")
print(f"State price: {state.price}")

# Call get_reward
reward = env.get_reward(state)

print(f"\nReward: {reward}")
print(f"Is reward close to 1214? {1200 < reward < 1600}")

# Check curriculum learning stage
if env.episode_number >= 150:
    print(f"Curriculum: HARD (no_trade_penalty: {env.no_trade_penalty})")
elif env.episode_number >= 50:
    print(f"Curriculum: MEDIUM (no_trade_penalty: {env.no_trade_penalty})")
else:
    print(f"Curriculum: EASY (no_trade_penalty: {env.no_trade_penalty})")

# Test if there's an issue with return values
print("\n\nTesting if there's a calculation issue...")
if hasattr(env, 'no_trade_penalty'):
    print(f"No trade penalty: {env.no_trade_penalty}")
else:
    print("No trade penalty not set!")
    
# Check if there might be a calculation path we're missing
print(f"\nChecking conditions:")
print(f"  current_position == 'FLAT': {env.current_position == 'FLAT'}")
print(f"  last_position == 'FLAT': {env.last_position == 'FLAT'}")
print(f"  trades_this_episode == 0: {env.trades_this_episode == 0}")
print(f"  current_index > 0: {env.current_index > 0}")