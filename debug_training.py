#!/usr/bin/env python3
"""Simple training script to debug the 11735 reward bug"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from futures_env_realistic import RealisticFuturesEnv
from rl_algorithms.ane_ppo import ANE_PPO
from data_manager import DataManager
import numpy as np
import logging

# Set up logging to see all our debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

print("Loading data...")
data_manager = DataManager()
states = data_manager.load_data('./data/processed/NQ_train_processed.csv')

if not states:
    print("No data loaded! Creating minimal test states...")
    # Create minimal states for testing
    from gym_futures.envs.utils import TimeSeriesState
    states = []
    for i in range(300):
        price = 3350 + i * 0.5
        state = TimeSeriesState(
            data=[[f'2008-01-{i//24+1:02d} {i%24:02d}:00:00', price-5, price+5, price-3, price, 1000]],
            close_price_identifier=4,
            timestamp_identifier=0
        )
        states.append(state)

print(f"Loaded {len(states)} states")

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

# Create algorithm
algorithm = ANE_PPO(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

print("\nRunning training to reproduce bug...")
print("Watch for these key debug messages:")
print("- GET_REWARD CALLED")
print("- STEP REWARD TRACE") 
print("- NO TRADE PENALTY")
print("- REWARD SCALING")
print("- HUGE HOLD REWARD")
print("-" * 50)

# Run enough episodes to trigger the bug
for episode in range(30):
    state = env.reset()
    env.episode_number = episode  # Set episode number for debugging
    done = False
    episode_reward = 0
    step = 0
    
    while not done and step < 200:
        # Take random actions for debugging
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        state = next_state
        step += 1
        
        # Extra logging for problematic episodes
        if abs(episode_reward) > 10000:
            print(f"\n!!! HUGE EPISODE REWARD DETECTED !!!")
            print(f"Episode {episode}, Step {step}: episode_reward={episode_reward:.2f}")
            print(f"Last step reward: {reward:.2f}")
            print(f"Trades this episode: {env.trades_this_episode}")
            print(f"Current position: {env.current_position}")
            break
    
    print(f"Episode {episode:2d} | Reward: {episode_reward:8.2f} | Trades: {env.trades_this_episode} | Steps: {step}")
    
    # Stop if we see the bug
    if abs(episode_reward) > 10000:
        print("\nBUG REPRODUCED! Check the logs above for debug messages.")
        break

print("\nDone. Check the output above for debug messages.")