#!/usr/bin/env python3
"""Test if step() is returning the correct values"""

import numpy as np
from futures_env_realistic import RealisticFuturesEnv
from train_standalone import SimpleDataLoader

# Load some test data
loader = SimpleDataLoader("data/processed/NQ_train_processed.csv") 
states = loader.load_time_series_states(limit=100)

if states:
    # Create environment
    env = RealisticFuturesEnv(states=states)
    
    # Create a wrapper to intercept step() return values
    original_step = env.step
    
    def debug_step(action):
        # Call original step
        result = original_step(action)
        
        # Unpack result
        obs, reward, done, info = result
        
        # Debug output
        if env.episode_number >= 60 and env.trades_this_episode == 0:
            print(f"\n=== STEP DEBUG (Episode {env.episode_number}) ===")
            print(f"Action: {action}")
            print(f"Trades: {env.trades_this_episode}")
            print(f"Position: {env.current_position}")
            print(f"Reward returned: {reward}")
            print(f"Observation shape: {obs.shape}")
            print(f"Observation sum: {np.sum(obs):.2f}")
            print(f"Observation mean: {np.mean(obs):.2f}")
            
            # Check if any observation value is close to 1214
            for i, val in enumerate(obs):
                if 1200 < val < 1600:
                    print(f"  WARNING: obs[{i}] = {val:.2f} is in suspicious range!")
                    
            # Check if observation sum or mean could be misinterpreted as reward
            if 1200 < np.sum(obs) < 1600:
                print(f"  WARNING: Observation sum ({np.sum(obs):.2f}) is in reward range!")
            if 1200 < np.mean(obs) < 1600:
                print(f"  WARNING: Observation mean ({np.mean(obs):.2f}) is in reward range!")
                
            # Check the actual values being returned
            print(f"\nActual return values:")
            print(f"  result[0] (obs) type: {type(obs)}, shape: {obs.shape}")
            print(f"  result[1] (reward) type: {type(reward)}, value: {reward}")
            print(f"  result[2] (done) type: {type(done)}, value: {done}")
            print(f"  result[3] (info) type: {type(info)}, value: {info}")
            
        return result
    
    # Replace step method
    env.step = debug_step
    
    # Set to problematic episode
    env.episode_number = 61
    
    # Reset and run a few steps
    print("Starting test at episode 61...")
    obs = env.reset()
    
    for i in range(5):
        action = 1  # HOLD
        obs, reward, done, info = env.step(action)
        
        if reward > 100:
            print(f"\n*** FOUND BUG: Step {i} returned reward = {reward} ***")
            break
else:
    print("No data loaded")