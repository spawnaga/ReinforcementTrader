#!/usr/bin/env python3
"""Simple debug script to check timestamp issues"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Check if we have any CSV files with data
data_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.csv') and 'data' in root and 'cache' not in root:
            data_files.append(os.path.join(root, file))

print(f"Found {len(data_files)} data files:")
for f in data_files[:5]:
    print(f"  - {f}")

# Check market data from database
print("\nChecking database data...")
try:
    from app import db
    from models import MarketData
    
    # Get a few market data records
    records = db.session.query(MarketData).limit(5).all()
    print(f"\nFound {len(records)} market data records in database")
    
    for i, record in enumerate(records):
        print(f"\nRecord {i}:")
        print(f"  timestamp: {record.timestamp} (type: {type(record.timestamp)})")
        print(f"  close_price: {record.close_price}")
        
except Exception as e:
    print(f"Error accessing database: {e}")

# Create a simple test dataframe
print("\n\nCreating test data with historical timestamps...")
test_data = pd.DataFrame({
    'timestamp': pd.date_range('2008-01-01', periods=100, freq='1min'),
    'open': 3600 + np.random.randn(100) * 10,
    'high': 3610 + np.random.randn(100) * 10,
    'low': 3590 + np.random.randn(100) * 10,
    'close': 3600 + np.random.randn(100) * 10,
    'volume': np.random.randint(100, 1000, 100)
})

print(f"Test data shape: {test_data.shape}")
print(f"First timestamp: {test_data['timestamp'].iloc[0]}")
print(f"Last timestamp: {test_data['timestamp'].iloc[-1]}")

# Test TimeSeriesState
from gym_futures.envs.utils import TimeSeriesState

window = test_data.iloc[0:50].copy()
state = TimeSeriesState(
    data=window,
    close_price_identifier='close',
    timestamp_identifier='timestamp'
)

print(f"\nTimeSeriesState created:")
print(f"  state.ts: {state.ts} (type: {type(state.ts)})")
print(f"  state.price: {state.price}")

# Check if it's current time
now = datetime.now()
if state.ts.date() == now.date():
    print(f"\n⚠️  WARNING: TimeSeriesState timestamp is TODAY ({now.date()})!")
    print(f"    This means historical timestamps are being converted to current time!")
else:
    print(f"\n✓ Good: TimeSeriesState timestamp is historical ({state.ts.date()})")