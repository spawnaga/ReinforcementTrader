#!/usr/bin/env python3
"""
Debug script to find the source of the 11,725 reward bug
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add current directory to path
sys.path.append('.')

from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState

def create_test_states():
    """Create test states with prices around 3350"""
    states = []
    base_price = 3350.0
    
    for i in range(200):
        # Price varies slightly around 3350
        price = base_price + np.random.uniform(-10, 10)
        
        # Create a data array with timestamp and OHLCV data
        # Format: [timestamp, open, high, low, close, volume]
        data = np.array([[
            datetime.now(),
            price - 2,      # open
            price + 2,      # high
            price - 3,      # low
            price,          # close
            1000           # volume
        ]])
        
        state = TimeSeriesState(
            data=data,
            close_price_identifier=4,  # close is at index 4
            timestamp_identifier=0     # timestamp is at index 0
        )
        states.append(state)
    
    return states

def test_reward_bug():
    """Test to reproduce the 11,725 reward bug"""
    
    print("Testing for 11,725 reward bug...")
    print("="*60)
    
    # Create environment with test parameters
    states = create_test_states()
    
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=3.5,  # This is key! 3350 * 3.5 = 11,725
        tick_size=0.25,
        execution_cost_per_order=5.0,
        session_id="debug_test",
        enable_trading_logger=True
    )
    
    # Test different episodes
    for episode in range(55):
        # Set episode number directly to test curriculum stages
        env.episode_number = episode
        
        obs = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        # Take a few steps without trading (action=1 is HOLD)
        while not done and step < 10:
            next_obs, reward, done, info = env.step(1)  # HOLD action
            
            # Check if we get the suspicious reward
            if abs(reward) > 1000:
                print(f"\n*** FOUND HUGE REWARD ***")
                print(f"Episode: {episode}")
                print(f"Step: {step}")
                print(f"Reward: {reward:.2f}")
                print(f"Trades this episode: {env.trades_this_episode}")
                print(f"Current position: {env.current_position}")
                print(f"Current price: {env.current_price:.2f}")
                
                # Check if reward matches our pattern
                if abs(reward - 11725) < 100:
                    print(f"\n*** THIS IS THE BUG! Reward ~11,725 ***")
                    print(f"Price * value_per_tick = {env.current_price} * {env.value_per_tick} = {env.current_price * env.value_per_tick:.2f}")
                    
                    # Check observation values
                    print(f"\nObservation shape: {next_obs.shape}")
                    print(f"Observation sum: {np.sum(next_obs):.2f}")
                    print(f"Observation mean: {np.mean(next_obs):.2f}")
                    print(f"First 5 obs values: {next_obs[:5]}")
                    
                    # Check if any observation value is close to reward
                    for i, val in enumerate(next_obs):
                        if abs(val - reward) < 100:
                            print(f"\n*** Observation[{i}] = {val:.2f} matches reward! ***")
                
                return True
            
            total_reward += reward
            step += 1
            obs = next_obs
        
        # Only print episodes where we don't trade
        if env.trades_this_episode == 0 and episode > 45:
            print(f"Episode {episode}: Total reward = {total_reward:.2f} (no trades)")
    
    print("\nBug not found in test")
    return False

def check_observation_calculation():
    """Check what values are in observations"""
    states = create_test_states()
    
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=3.5,
        tick_size=0.25,
        execution_cost_per_order=5.0,
        session_id="debug_test",
        enable_trading_logger=False
    )
    
    env.reset()
    obs = env._get_observation(states[0])
    
    print("\nChecking observation values:")
    print(f"Observation shape: {obs.shape}")
    print(f"Observation sum: {np.sum(obs):.2f}")
    print(f"Observation mean: {np.mean(obs):.2f}")
    print(f"Max observation value: {np.max(obs):.2f}")
    print(f"Min observation value: {np.min(obs):.2f}")
    
    # Check if any value is around 11,725
    for i, val in enumerate(obs):
        if val > 10000:
            print(f"\nLarge value found in observation[{i}]: {val:.2f}")
            
            # Check if it's price * value_per_tick
            if abs(val / env.value_per_tick - 3350) < 100:
                print(f"*** This is price * value_per_tick! ***")
                print(f"{val:.2f} / {env.value_per_tick} = {val / env.value_per_tick:.2f}")

if __name__ == "__main__":
    print("Debug script for 11,725 reward bug")
    print("="*60)
    
    # First check observation values
    check_observation_calculation()
    
    print("\n" + "="*60)
    
    # Then test for the bug
    if test_reward_bug():
        print("\nBug reproduced successfully!")
    else:
        print("\nCould not reproduce bug with test data")
    
    print("\nNote: The bug might require specific market data or conditions")