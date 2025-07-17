#!/usr/bin/env python3
"""Debug script to find why episodes with 0 trades get -1283 rewards"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState

# Create minimal test states
states = []
for i in range(250):
    price = 3350 + i * 0.5
    state = TimeSeriesState(
        data=[[f'2008-01-{i//24+1:02d} {i%24:02d}:00:00', price-5, price+5, price-3, price, 1000]],
        close_price_identifier=4,
        timestamp_identifier=0
    )
    states.append(state)

# Create environment
env = RealisticFuturesEnv(
    states=states,
    episode_len=200,
    tick_size=0.25,
    value_per_tick=5.0,
    commission=5.0,
    min_holding_periods=10,
    max_trades_per_episode=10
)

# Test an episode where agent never trades
print("Testing episode with no trades...")
state = env.reset()

# Set episode number to trigger the issue
env.episode_number = 27

total_reward = 0
rewards_per_step = []

# Take only HOLD actions (action=1) for entire episode
for step in range(200):
    action = 1  # Always HOLD
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    rewards_per_step.append(reward)
    
    # Print debug info for key steps
    if step < 5 or step == 50 or step == 100 or step == 150 or step >= 195:
        print(f"Step {step}: reward={reward:.4f}, cumulative={total_reward:.4f}, trades={env.trades_this_episode}")
    
    if done:
        break

print(f"\nTotal episode reward: {total_reward:.2f}")
print(f"Average reward per step: {total_reward/200:.4f}")
print(f"Number of trades: {env.trades_this_episode}")

# Analyze the rewards
unique_rewards = list(set(rewards_per_step))
print(f"\nUnique reward values: {unique_rewards}")

# Count penalty steps
penalty_steps = sum(1 for r in rewards_per_step if r < 0)
print(f"Steps with penalties: {penalty_steps}")

# Check if it matches the -1283 pattern
if abs(total_reward + 1283) < 20:
    print("\n*** FOUND THE BUG! ***")
    print(f"Total reward {total_reward:.2f} matches the -1283 pattern")
    print(f"This suggests a per-step penalty of: {total_reward/200:.4f}")