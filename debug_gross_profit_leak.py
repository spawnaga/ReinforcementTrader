#!/usr/bin/env python3
"""
Debug script to find where gross profit from previous trades leaks into rewards
"""

import sys
import logging
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.DEBUG)

print("Creating test environment to reproduce gross profit leak...")

# Create test states with specific prices to reproduce the bug
base_time = datetime(2021, 6, 1, 9, 0, 0)
prices = [3354.25, 3350.00, 3345.00, 3340.00, 3335.00, 3330.00, 3325.00, 3320.00, 
          3315.00, 3310.00, 3305.00, 3300.00, 3295.00, 3290.00, 3285.00, 3284.50]  # Last price gives ~279 ticks profit

states = []
for i, price in enumerate(prices):
    state = TimeSeriesState(
        timestamp=base_time + timedelta(minutes=i*5),
        data=pd.Series({'price': price}),
        price=price
    )
    states.append(state)

# Add more states for after the trade
for i in range(20):
    price = 3285.00 + i * 5  # Prices going back up
    state = TimeSeriesState(
        timestamp=base_time + timedelta(minutes=(len(prices) + i)*5),
        data=pd.Series({'price': price}),
        price=price
    )
    states.append(state)

# Create environment
env = RealisticFuturesEnv(states=states, episode_number=9)
env.trading_logger = None  # Disable logging for clarity

print(f"\nTest scenario: Opening SHORT at ${prices[0]}, closing at ${prices[-1]}")
print(f"Expected gross profit: ({prices[0]} - {prices[-1]}) / 0.25 * $5 = ${(prices[0] - prices[-1]) / 0.25 * 5}")
print(f"Expected net profit after $10 costs: ${(prices[0] - prices[-1]) / 0.25 * 5 - 10}")

# Reset environment
obs = env.reset()

print("\n=== STEP BY STEP EXECUTION ===\n")

# Step 1: Open SHORT position
print("Step 1: Opening SHORT position")
obs, reward, done, _ = env.step(2)  # SELL action
print(f"  Position after: {env.current_position}")
print(f"  Entry price: {env.entry_price}")
print(f"  Reward: ${reward}")

# Step through holding the position
for i in range(14):
    print(f"\nStep {i+2}: Holding position")
    obs, reward, done, _ = env.step(1)  # HOLD action
    print(f"  Current price: {states[i+1].price}")
    print(f"  Reward: ${reward}")

# Close the SHORT position by buying
print(f"\nStep 16: Closing SHORT position")
obs, reward, done, _ = env.step(0)  # BUY action
print(f"  Position after: {env.current_position}")
print(f"  Exit price: {env.exit_price if hasattr(env, 'exit_price') else 'N/A'}")
print(f"  Reward: ${reward}")
print(f"  *** This should be the NET profit after costs ***")

# Calculate what the trade profit was
if env._last_closed_entry_price and env._last_closed_exit_price:
    gross_ticks = (env._last_closed_entry_price - env._last_closed_exit_price) / 0.25
    gross_profit = gross_ticks * 5
    net_profit = gross_profit - 10
    print(f"\n  Trade calculation:")
    print(f"    Gross ticks: {gross_ticks}")
    print(f"    Gross profit: ${gross_profit}")
    print(f"    Net profit: ${net_profit}")

# Now open a new position - THIS IS WHERE THE BUG HAPPENS
print(f"\n=== CRITICAL TEST: Opening new position after trade closes ===")
print(f"Step 17: Opening new LONG position")
obs, reward, done, _ = env.step(0)  # BUY action
print(f"  Position after: {env.current_position}")
print(f"  Entry price: {env.entry_price}")
print(f"  Reward: ${reward}")
print(f"  *** BUG: If reward = ${gross_profit}, then gross profit is leaking! ***")

if abs(reward - gross_profit) < 1:
    print(f"\n!!! BUG CONFIRMED !!!")
    print(f"Reward ({reward}) matches gross profit ({gross_profit}) from previous trade!")
    print(f"The environment is returning the previous trade's gross profit as reward!")

# Check environment state
print(f"\n=== ENVIRONMENT STATE CHECK ===")
attrs_to_check = ['total_net_profit', 'total_reward', '_last_closed_entry_price', '_last_closed_exit_price']
for attr in attrs_to_check:
    if hasattr(env, attr):
        value = getattr(env, attr)
        print(f"{attr}: {value}")