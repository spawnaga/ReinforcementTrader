#!/usr/bin/env python3
"""Debug script to find where the 1214 reward is coming from"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState
import pandas as pd

# Create realistic states with prices around 3350
states = []
for i in range(100):
    price = 3350 + i * 0.25
    # Create TimeSeriesState with proper structure
    data = pd.DataFrame({
        'timestamp': [pd.Timestamp(f'2008-01-02 {8+i//60:02d}:{i%60:02d}:00')],
        'open': [price - 2],
        'high': [price + 2],
        'low': [price - 3],
        'close': [price],
        'volume': [1000 + i * 10]
    })
    
    state = TimeSeriesState(
        data=data.values,
        close_price_identifier=4,
        timestamp_identifier=0,
        timestamp_format='%Y-%m-%d %H:%M:%S'
    )
    states.append(state)

# Create environment
env = RealisticFuturesEnv(
    states=states,
    tick_size=0.25,
    value_per_tick=5.0
)

# Create a mock trading logger to avoid attribute errors
class MockLogger:
    def warning(self, msg): pass
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): pass
    def log_state_debug(self, type, msg): pass
    def log_error(self, type, msg, context=None): pass
    def log_reward_calculation(self, *args, **kwargs): pass

env.trading_logger = MockLogger()

# Set environment to problematic state
env.episode_number = 61
env._apply_curriculum_learning()

# Reset and check initial state
obs = env.reset()
print(f"After reset:")
print(f"  Episode: {env.episode_number}")
print(f"  Trades: {env.trades_this_episode}")
print(f"  Current position: {env.current_position}")
print(f"  No trade penalty: {env.no_trade_penalty}")
print(f"  Observation shape: {obs.shape}")

# Take a few HOLD actions to see what happens
print("\nTaking HOLD actions:")
for i in range(5):
    action = 1  # HOLD
    obs, reward, done, info = env.step(action)
    
    print(f"\nStep {i}:")
    print(f"  Reward: {reward}")
    print(f"  Trades: {env.trades_this_episode}")
    print(f"  Position: {env.current_position}")
    print(f"  Current price: {env.states[env.current_index].price if env.current_index < len(env.states) else 'N/A'}")
    
    # Check if reward is in suspicious range
    if 1200 < reward < 1600:
        print(f"  *** SUSPICIOUS REWARD DETECTED: {reward} ***")
        print(f"  Price / 2.75 = {env.states[env.current_index].price / 2.75}")
        
    # Check observation sum
    obs_sum = np.sum(obs)
    if 1200 < obs_sum < 1600:
        print(f"  *** OBSERVATION SUM IN RANGE: {obs_sum} ***")

# Now let's trace through get_reward manually
print("\n\nManually calling get_reward:")
if env.current_index < len(env.states):
    state = env.states[env.current_index]
    
    # Temporarily override some values to trace the issue
    env.trades_this_episode = 0
    env.current_position = "FLAT"
    env.last_position = "FLAT"
    
    # Add debugging to get_reward
    print(f"\nCalling get_reward with:")
    print(f"  State price: {state.price}")
    print(f"  Episode: {env.episode_number}")
    print(f"  Trades: {env.trades_this_episode}")
    print(f"  Position: {env.current_position}")
    
    reward = env.get_reward(state)
    print(f"\nget_reward returned: {reward}")
    print(f"Is reward close to price/2.75? {abs(reward - state.price/2.75) < 1}")