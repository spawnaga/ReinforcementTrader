#!/usr/bin/env python3
"""Quick test to catch the 1214 bug in action"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from train_standalone import SimpleDataLoader, create_time_series_states
from futures_env_realistic import RealisticFuturesEnv
from rl_algorithms.ane_ppo import ANE_PPO
import torch
import numpy as np

# Load minimal data
loader = SimpleDataLoader()
df = loader.load_csv("data/processed/NQ_train_processed.csv")
if df is None or df.empty:
    print("No data loaded")
    exit(1)

# Create states
states = create_time_series_states(df, limit=200)
print(f"Loaded {len(states)} states")

# Create environment
env = RealisticFuturesEnv(states=states)

# Create a simple ANE-PPO algorithm
state_dim = env.observation_space.shape[0]
action_dim = 3  # BUY, HOLD, SELL
algorithm = ANE_PPO(state_dim, action_dim)

# Set episode to 61 to trigger bug
env.episode_number = 61

# Reset environment
state = env.reset()
print(f"\nStarting at episode {env.episode_number}")
print(f"Initial state shape: {state.shape}")

# Run a few steps
for step in range(10):
    # Get action from algorithm
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        action_probs, _ = algorithm.actor_critic(state_tensor)
        action = torch.multinomial(action_probs[0], 1).item()
    
    # Take step
    next_state, reward, done, info = env.step(action)
    
    # Check for the bug
    if reward > 100:
        print(f"\n*** BUG DETECTED at step {step} ***")
        print(f"Episode: {env.episode_number}")
        print(f"Action: {['BUY', 'HOLD', 'SELL'][action]}")
        print(f"Reward: {reward}")
        print(f"Trades: {env.trades_this_episode}")
        print(f"Position: {env.current_position}")
        print(f"Current price: {env.states[env.current_index].price if env.current_index < len(env.states) else 'N/A'}")
        
        # Check if reward is price/2.75
        if env.current_index < len(env.states):
            price = env.states[env.current_index].price
            if abs(reward - price/2.75) < 1:
                print(f"\n*** CONFIRMED: reward ({reward:.2f}) = price ({price:.2f}) / 2.75 ***")
                print(f"tick_size = {env.tick_size}")
                print(f"tick_size * 11 = {env.tick_size * 11}")
        break
    
    state = next_state
    
    if done:
        print(f"Episode ended at step {step}")
        break

print("\nTest completed")