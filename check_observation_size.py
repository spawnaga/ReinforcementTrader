#!/usr/bin/env python3
"""Check the observation size and see if it relates to the 1214 bug"""

import numpy as np
from train_standalone import SimpleDataLoader
from gym_futures.envs import RealisticFuturesEnv

# Load data using the same loader as training
loader = SimpleDataLoader("data/processed/NQ_train_processed.csv")
states = loader.load_time_series_states(limit=100)

if states:
    # Check the flattened size of a state
    state = states[0]
    
    # Check what flatten() returns
    if hasattr(state, 'data'):
        print(f"State data shape: {np.array(state.data).shape}")
        flat = np.array(state.data).flatten()
        print(f"Flattened shape: {flat.shape}")
        print(f"Number of features: {len(flat)}")
        print(f"First 10 values: {flat[:10]}")
        
        # Check if any division could produce 1214
        if state.price:
            print(f"\nState price: {state.price}")
            for i in range(1, 20):
                result = state.price / (0.25 * i)
                if 1200 < result < 1300:
                    print(f"  Price / (0.25 * {i}) = {result:.2f}")
                    
    # Create environment and check observation
    env = RealisticFuturesEnv(states=states[:10])
    obs = env.reset()
    print(f"\nEnvironment observation shape: {obs.shape}")
    print(f"Observation size: {len(obs)}")
    print(f"First 10 obs values: {obs[:10]}")
    print(f"Observation sum: {np.sum(obs):.2f}")
    
    # Check if sum/size gives us something close to 1214
    if len(obs) > 0:
        print(f"Observation sum / size = {np.sum(obs) / len(obs):.2f}")
        
        # Check if any element is close to 1214
        for i, val in enumerate(obs):
            if 1200 < val < 1600:
                print(f"  obs[{i}] = {val:.2f} (SUSPICIOUS!)")
else:
    print("No states loaded")