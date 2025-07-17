#!/usr/bin/env python3
"""Test timestamp parsing from the user's data"""

import pandas as pd
import numpy as np
from datetime import datetime
from gym_futures.envs.utils import TimeSeriesState

# Sample data from user
sample_line = "1622493300000000000,14880,14880.75,14878,14879,73,20,35,0,-0.7798844830928816,0.6259234721840593,0,1,14873.275,14874.72833908955,4311803"
values = sample_line.split(',')

# Parse timestamp
timestamp_ns = int(values[0])
print(f"Timestamp in nanoseconds: {timestamp_ns}")

# Convert nanoseconds to datetime
timestamp_s = timestamp_ns / 1e9  # Convert to seconds
timestamp_dt = pd.Timestamp(timestamp_s, unit='s')
print(f"Converted timestamp: {timestamp_dt}")
print(f"Date: {timestamp_dt.date()}")
print(f"Time: {timestamp_dt.time()}")

# Create a small test dataframe
test_data = pd.DataFrame({
    'timestamp': [1622493300000000000, 1622493360000000000, 1622493420000000000],
    'open': [14880, 14879.25, 14879.25],
    'high': [14880.75, 14879.5, 14882.75],
    'low': [14878, 14878.5, 14878.5],
    'close': [14879, 14879, 14880.5],
    'volume': [73, 30, 184]
})

print("\nTest DataFrame:")
print(test_data.head())

# Convert timestamp column to datetime
test_data['timestamp'] = pd.to_datetime(test_data['timestamp'], unit='ns')
print("\nAfter timestamp conversion:")
print(test_data.head())

# Test TimeSeriesState
print("\nTesting TimeSeriesState...")
state = TimeSeriesState(
    data=test_data,
    close_price_identifier='close',
    timestamp_identifier='timestamp'
)

print(f"State timestamp: {state.ts}")
print(f"State price: {state.price}")

# Check if it's being converted to current time
now = datetime.now()
if state.ts.date() == now.date():
    print(f"\n⚠️  ERROR: TimeSeriesState timestamp is TODAY ({now.date()})!")
    print(f"    Historical timestamp was converted to current time!")
else:
    print(f"\n✓ SUCCESS: TimeSeriesState timestamp is historical ({state.ts.date()})")