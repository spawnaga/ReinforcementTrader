#!/usr/bin/env python3
"""
Test to confirm the reward leak bug - gross profit from previous trade 
appears as reward when opening new position
"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState
import pandas as pd
from datetime import datetime, timedelta

# Create simple test data
base_time = datetime(2021, 6, 1, 9, 0, 0)
data = []

# Create price data that will give us a specific profit
# Start at 3300, drop to 3280 (80 ticks = $400 gross profit for short)
for i in range(20):
    price = 3300 - i * 1  # Drop by 1 point each step
    row = pd.DataFrame({
        'timestamp': [base_time + timedelta(minutes=i*5)],
        'open': [price],
        'high': [price + 0.5],
        'low': [price - 0.5],
        'close': [price],
        'volume': [1000],
        'price': [price]
    })
    data.append(row)

# Add more data for after the trade
for i in range(20):
    price = 3280 + i * 0.5  # Price goes back up
    row = pd.DataFrame({
        'timestamp': [base_time + timedelta(minutes=(20+i)*5)],
        'open': [price],
        'high': [price + 0.5],
        'low': [price - 0.5],
        'close': [price],
        'volume': [1000],
        'price': [price]
    })
    data.append(row)

df = pd.concat(data, ignore_index=True)

# Create states
states = []
for idx, row in df.iterrows():
    state = TimeSeriesState(
        timestamp_format='%Y-%m-%d %H:%M:%S',
        data=row
    )
    state.ts = row['timestamp']
    state.price = row['price']
    states.append(state)

# Create environment
print("Creating test environment...")
env = RealisticFuturesEnv(
    states=states,
    episode_number=9,  # Episode where bug appears
    value_per_tick=5,
    execution_cost_per_order=5
)

# Reset
obs = env.reset()
print(f"\nStarting test - Episode {env.episode_number}")
print("Expected gross profit for 80 tick move: 80 × $5 = $400")
print("Expected net profit after $10 costs: $400 - $10 = $390")

# Track all rewards
all_rewards = []

print("\n=== OPENING SHORT POSITION ===")
# Step 1: Open SHORT
obs, reward, done, _ = env.step(2)  # SELL
all_rewards.append(reward)
print(f"Step 1: Action=SELL, Reward=${reward:.2f}, Position={env.current_position}")

# Hold for several steps
print("\n=== HOLDING POSITION ===")
for i in range(18):
    obs, reward, done, _ = env.step(1)  # HOLD
    all_rewards.append(reward)
    if i % 5 == 0:
        print(f"Step {i+2}: Action=HOLD, Reward=${reward:.2f}, Price=${states[i+1].price:.2f}")

print("\n=== CLOSING SHORT POSITION ===")
# Close SHORT by buying
obs, reward, done, _ = env.step(0)  # BUY to close short
all_rewards.append(reward)
print(f"Step 20: Action=BUY (close short), Reward=${reward:.2f}, Position={env.current_position}")

# Calculate what the gross profit was
if env._last_closed_entry_price and env._last_closed_exit_price:
    gross_ticks = (env._last_closed_entry_price - env._last_closed_exit_price) / 0.25
    gross_profit = gross_ticks * 5
    net_profit = gross_profit - 10
    print(f"\nTrade Results:")
    print(f"  Entry: ${env._last_closed_entry_price:.2f}")
    print(f"  Exit: ${env._last_closed_exit_price:.2f}")
    print(f"  Gross ticks: {gross_ticks:.1f}")
    print(f"  Gross profit: ${gross_profit:.2f}")
    print(f"  Net profit: ${net_profit:.2f}")
    print(f"  Actual reward received: ${reward:.2f}")

print("\n=== CRITICAL TEST: OPENING NEW POSITION ===")
# This is where the bug happens - open a new position
obs, reward, done, _ = env.step(0)  # BUY to open long
all_rewards.append(reward)
print(f"Step 21: Action=BUY (open long), Reward=${reward:.2f}, Position={env.current_position}")

if abs(reward - gross_profit) < 1:
    print(f"\n🚨 BUG CONFIRMED! 🚨")
    print(f"Reward (${reward:.2f}) = Gross profit (${gross_profit:.2f}) from previous trade!")
    print(f"The environment is leaking the previous trade's gross profit into the reward!")
else:
    print(f"\nReward doesn't match gross profit. Bug might be elsewhere.")

# Check a few more steps
print("\n=== SUBSEQUENT STEPS ===")
for i in range(3):
    obs, reward, done, _ = env.step(1)  # HOLD
    all_rewards.append(reward)
    print(f"Step {22+i}: Action=HOLD, Reward=${reward:.2f}")

print(f"\n=== SUMMARY ===")
print(f"Total rewards collected: {len(all_rewards)}")
print(f"Sum of all rewards: ${sum(all_rewards):.2f}")
print(f"Rewards > $100: {[r for r in all_rewards if abs(r) > 100]}")