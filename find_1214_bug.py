#!/usr/bin/env python3
"""Find the exact source of the 1214 bug"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState

# Create test states with price = 3350 (since 3350 / 2.75 = 1218.18)
states = []
for i in range(200):
    state = TimeSeriesState(
        price=3350.0,
        volume=1000,
        adx=25.0,
        rsi=50.0,
        macd=0.0,
        macd_signal=0.0,
        macd_diff=0.0,
        bb_upper=3400.0,
        bb_lower=3300.0,
        atr=50.0,
        time_value=np.array([0.5, 0.5, 0.5, 0.5, 2, 12]),  # sin_time, cos_time, etc.
        pct_change=0.001,
        pct_change_vol=0.0,
        ema_fast=3350.0,
        ema_medium=3350.0,
        ema_slow=3350.0,
        ts="2024-01-01"
    )
    states.append(state)

# Create environment
env = RealisticFuturesEnv(states=states)

# Test different episode numbers
for episode in [59, 60, 61, 62]:
    print(f"\n=== Testing Episode {episode} ===")
    env.episode_number = episode
    env.trades_this_episode = 0
    env.current_position = 0  # FLAT
    env.last_position = 0
    env.entry_price = None
    env.current_index = 100  # Past the 50 step threshold
    
    # Get reward
    reward = env.get_reward(states[100])
    print(f"Episode {episode}: Reward = {reward}")
    
    # Check if reward matches our bug pattern
    if abs(reward) > 100:
        print(f"*** BUG DETECTED! ***")
        price_div_2_75 = 3350.0 / 2.75
        print(f"Price / 2.75 = {price_div_2_75:.2f}")
        print(f"Reward = {reward:.2f}")
        print(f"Match? {abs(reward - price_div_2_75) < 1}")
        
        # Check what 2.75 represents
        print(f"\nInvestigating 2.75:")
        print(f"tick_size = {env.tick_size}")
        print(f"tick_size * 11 = {env.tick_size * 11}")
        print(f"value_per_tick = {env.value_per_tick}")

# Test with different prices to confirm pattern
print("\n=== Testing different prices ===")
test_prices = [3300, 3400, 3500]
for price in test_prices:
    states[0].price = price
    env.episode_number = 61
    env.trades_this_episode = 0
    env.current_position = 0
    env.last_position = 0
    env.entry_price = None
    env.current_index = 100
    
    reward = env.get_reward(states[0])
    expected = price / 2.75
    
    print(f"Price {price}: Reward = {reward:.2f}, Expected (price/2.75) = {expected:.2f}")