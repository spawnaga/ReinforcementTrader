#!/usr/bin/env python3
"""Debug timestamp issues in training data"""

import pandas as pd
from train_standalone import SimpleDataLoader
from gym_futures.envs.utils import TimeSeriesState

def debug_timestamps():
    # Load the data
    loader = SimpleDataLoader()
    df = loader.load_csv('data/processed/NQ_continuous_adjusted_1min_trading_hours_EST.csv')
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check timestamp column
    if 'timestamp' in df.columns:
        print(f"\nTimestamp column found!")
        print(f"First 5 timestamps:")
        for i in range(min(5, len(df))):
            print(f"  {i}: {df['timestamp'].iloc[i]} (type: {type(df['timestamp'].iloc[i])})")
        
        print(f"\nLast 5 timestamps:")
        for i in range(max(0, len(df)-5), len(df)):
            print(f"  {i}: {df['timestamp'].iloc[i]} (type: {type(df['timestamp'].iloc[i])})")
    else:
        print(f"\nNo timestamp column found! Available columns: {list(df.columns)}")
    
    # Create a sample TimeSeriesState
    if len(df) > 50:
        window_data = df.iloc[0:50].copy()
        print(f"\nCreating TimeSeriesState from first 50 rows...")
        
        state = TimeSeriesState(
            data=window_data,
            close_price_identifier='close',
            timestamp_identifier='timestamp'
        )
        
        print(f"State timestamp: {state.ts} (type: {type(state.ts)})")
        print(f"State price: {state.price}")
        
        # Check if timestamp is being converted to current time
        import datetime
        now = datetime.datetime.now()
        if state.ts.date() == now.date():
            print(f"\nWARNING: State timestamp is TODAY's date! Should be historical.")
        else:
            print(f"\nGood: State timestamp is historical: {state.ts.date()}")

if __name__ == "__main__":
    debug_timestamps()