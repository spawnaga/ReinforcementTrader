#!/usr/bin/env python3
"""
Trace reward calculations to find where gross profit leaks
"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState
import pandas as pd
from datetime import datetime, timedelta

# Monkey patch the get_reward method to add detailed logging
original_get_reward = RealisticFuturesEnv.get_reward

def traced_get_reward(self, state):
    print(f"\n=== GET_REWARD CALLED ===")
    print(f"  Current position: {self.current_position}")
    print(f"  Last position: {self.last_position}")
    print(f"  Episode: {self.episode_number}")
    
    if hasattr(self, '_last_closed_entry_price') and self._last_closed_entry_price:
        print(f"  Last closed entry: ${self._last_closed_entry_price}")
        print(f"  Last closed exit: ${self._last_closed_exit_price}")
        if self._last_closed_entry_price and self._last_closed_exit_price:
            gross_ticks = abs(self._last_closed_entry_price - self._last_closed_exit_price) / 0.25
            gross_profit = gross_ticks * 5
            print(f"  Previous trade gross profit: ${gross_profit}")
    
    # Call original method
    reward = original_get_reward(self, state)
    
    print(f"  RETURNING REWARD: ${reward}")
    
    # Check if reward matches gross profit
    if hasattr(self, '_last_closed_entry_price') and self._last_closed_entry_price and self._last_closed_exit_price:
        gross_ticks = abs(self._last_closed_entry_price - self._last_closed_exit_price) / 0.25
        gross_profit = gross_ticks * 5
        if abs(reward - gross_profit) < 1:
            print(f"  🚨 BUG: Reward matches gross profit from previous trade!")
    
    return reward

# Apply the patch
RealisticFuturesEnv.get_reward = traced_get_reward

# Create simple test data
data = []
base_time = datetime(2021, 6, 1, 9, 0, 0)

# Price drops from 3300 to 3220 (80 ticks for short trade)
prices = [3300 - i*2 for i in range(40)]

for i, price in enumerate(prices):
    data.append({
        'timestamp': base_time + timedelta(minutes=i*5),
        'open': price,
        'high': price + 1,
        'low': price - 1,
        'close': price,
        'volume': 1000
    })

df = pd.DataFrame(data)
states = []
for _, row in df.iterrows():
    state = TimeSeriesState(data=row, timestamp_format='%Y-%m-%d %H:%M:%S')
    state.ts = row['timestamp']
    state.price = row['close']
    states.append(state)

# Test with episode 9 where bug appears
env = RealisticFuturesEnv(states=states, episode_number=9)
obs = env.reset()

print("\n=== STARTING TEST - Episode 9 ===")
print("Opening SHORT position...")

# Open SHORT
obs, reward, done, _ = env.step(2)  # SELL
print(f"\nStep 1 result: Reward=${reward}")

# Hold for a while
print("\nHolding position...")
for i in range(10):
    obs, reward, done, _ = env.step(1)  # HOLD

# Close SHORT
print("\nClosing SHORT position...")
obs, reward, done, _ = env.step(0)  # BUY
print(f"\nClose trade result: Reward=${reward}")

# THE BUG: Open new position immediately
print("\n🔍 CRITICAL TEST: Opening new LONG position...")
obs, reward, done, _ = env.step(0)  # BUY
print(f"\nOpen new position result: Reward=${reward}")

if reward > 100:
    print("\n🚨 BUG CONFIRMED! Got huge reward when opening new position!")