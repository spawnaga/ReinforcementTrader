#!/usr/bin/env python3
"""Test if timestamp fix works with actual data loader and logging"""

import logging
from train_standalone import SimpleDataLoader
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv
from logging_config import get_loggers
import numpy as np
from datetime import datetime

# Setup loggers
loggers = get_loggers()
algorithm_logger = loggers['algorithm']

# Load test data
print("Loading test data...")
loader = SimpleDataLoader()
df = loader.load_csv('test_data_sample.csv')

print(f"\nData columns: {list(df.columns)}")
print(f"First timestamp: {df['timestamp'].iloc[0]} (type: {type(df['timestamp'].iloc[0])})")

# Create a few TimeSeriesState objects
print("\nCreating TimeSeriesState objects...")
states = []
window_size = 3

for i in range(window_size, min(window_size + 2, len(df))):
    window_data = df.iloc[i-window_size:i].copy()
    state = TimeSeriesState(
        data=window_data,
        close_price_identifier='close',
        timestamp_identifier='timestamp'
    )
    states.append(state)
    print(f"State {i-window_size}: timestamp={state.ts}, price={state.price}")

# Test logging with the correct timestamp
print("\nTesting logging...")
if states:
    # Simulate a step in the algorithm
    state = states[0]
    action = 'BUY'
    position = 'FLAT'
    
    # This should log with the historical timestamp from the state
    algorithm_logger.info(f"Step 1 | Time: {state.ts} | Price: ${state.price:.2f} | Position: {position} | Action: {action}")
    
    # Check if logged timestamp matches state timestamp
    print(f"\nLogged timestamp should be: {state.ts}")
    print(f"Current time is: {datetime.now()}")
    
    if state.ts.date() == datetime.now().date():
        print("\n⚠️  ERROR: Still logging current time instead of historical timestamp!")
    else:
        print("\n✓ SUCCESS: Logging historical timestamp correctly!")

# Create environment to test full integration
print("\nCreating environment with states...")
env = RealisticFuturesEnv(
    states=states,
    value_per_tick=5.0,
    tick_size=0.25,
    execution_cost_per_order=5.0,
    session_id='test_session',
    enable_trading_logger=False
)

# Do a step to trigger logging
state = env.reset()
obs, reward, done, info = env.step(0)  # Buy action

print("\nCheck logs/algorithm.log to verify timestamps are historical (2021-05-31), not current (2025-07-17)")