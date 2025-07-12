"""
Trading System Diagnostic Tool

This script checks the trading system to understand why entry/exit prices are None
and provides a comprehensive report of the trading state.
"""
import os
import sys
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_manager import DataManager
from gym_futures.envs.futures_env import FuturesEnv
from gym_futures.envs.utils import TimeSeriesState

def check_data_loading():
    """Check if data is loading correctly"""
    print("\n=== DATA LOADING CHECK ===")
    try:
        dm = DataManager()
        
        # Check available data files
        data_files = list(Path('data').glob('*.csv')) + list(Path('data').glob('*.txt'))
        print(f"Found {len(data_files)} data files:")
        for f in data_files[:5]:  # Show first 5
            print(f"  - {f.name}")
        
        if data_files:
            # Try to load the first file
            test_file = data_files[0]
            print(f"\nTesting data load from: {test_file}")
            
            df = dm.load_futures_data(
                symbol='NQ',
                filepath=str(test_file)
            )
            
            if df is not None and not df.empty:
                print(f"✓ Successfully loaded {len(df)} rows")
                print(f"✓ Columns: {list(df.columns)}")
                print(f"✓ Date range: {df.index[0]} to {df.index[-1]}")
                
                # Check for None/NaN values
                null_counts = df.isnull().sum()
                if null_counts.any():
                    print("\n⚠️  WARNING: Found null values:")
                    for col, count in null_counts[null_counts > 0].items():
                        print(f"    - {col}: {count} null values")
                else:
                    print("✓ No null values found in data")
                
                # Sample data
                print("\nSample data (first 5 rows):")
                print(df.head())
                
                return df
            else:
                print("✗ Failed to load data - empty dataframe")
                return None
        else:
            print("✗ No data files found in data directory")
            return None
            
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_timeseries_states(df):
    """Check if TimeSeriesState objects are created correctly"""
    print("\n=== TIMESERIES STATE CHECK ===")
    
    try:
        # Create a few sample states
        states = []
        window_size = 20
        
        for i in range(window_size, min(window_size + 10, len(df))):
            window_data = df.iloc[i-window_size:i]
            
            # Check if we have price data
            if 'close' in window_data.columns:
                current_price = window_data['close'].iloc[-1]
                timestamp = window_data.index[-1]
                
                state = TimeSeriesState(
                    data=window_data,
                    timestamp=timestamp,
                    price=current_price,
                    window=window_size
                )
                states.append(state)
                
                print(f"✓ Created state {i-window_size}: price={current_price:.2f}, timestamp={timestamp}")
            else:
                print(f"✗ Missing 'close' column in data")
                break
        
        if states:
            print(f"\n✓ Successfully created {len(states)} TimeSeriesState objects")
            print(f"✓ First state price: {states[0].price}")
            print(f"✓ Last state price: {states[-1].price}")
            return states
        else:
            print("✗ Failed to create any TimeSeriesState objects")
            return []
            
    except Exception as e:
        print(f"✗ Error creating TimeSeriesState objects: {e}")
        import traceback
        traceback.print_exc()
        return []

def check_trading_environment(states):
    """Check if the trading environment works correctly"""
    print("\n=== TRADING ENVIRONMENT CHECK ===")
    
    if not states:
        print("✗ No states available to test environment")
        return
    
    try:
        # Create environment
        env = FuturesEnv(
            states=states,
            value_per_tick=5.0,  # NQ value per tick
            tick_size=0.25,      # NQ tick size
            execution_cost_per_order=2.0,
            enable_trading_logger=True
        )
        
        print("✓ Environment created successfully")
        print(f"  - Number of states: {len(env.states)}")
        print(f"  - Value per tick: ${env.value_per_tick}")
        print(f"  - Tick size: {env.tick_size}")
        
        # Reset environment
        initial_state = env.reset()
        print(f"\n✓ Environment reset successful")
        if hasattr(initial_state, 'price'):
            print(f"  - Initial state price: {initial_state.price}")
        
        # Simulate a few trading steps
        print("\n=== SIMULATING TRADES ===")
        
        # Test buy action
        print("\n1. Testing BUY action...")
        state, reward, done, info = env.step(0)  # 0 = buy
        print(f"  - Position after buy: {env.current_position}")
        print(f"  - Entry price: {env.entry_price}")
        print(f"  - Reward: {reward}")
        print(f"  - Info: {info.get('message', 'No message')}")
        
        # Test hold action
        print("\n2. Testing HOLD action...")
        state, reward, done, info = env.step(1)  # 1 = hold
        print(f"  - Position after hold: {env.current_position}")
        print(f"  - Reward: {reward}")
        
        # Test sell action to close position
        print("\n3. Testing SELL action (close long)...")
        state, reward, done, info = env.step(2)  # 2 = sell
        print(f"  - Position after sell: {env.current_position}")
        print(f"  - Exit price: {env.exit_price}")
        print(f"  - Reward: {reward}")
        print(f"  - Info: {info.get('message', 'No message')}")
        
        # Check for None prices
        if env.entry_price is None:
            print("\n⚠️  WARNING: Entry price is None!")
        if env.exit_price is None and env.current_position == 0:
            print("⚠️  WARNING: Exit price is None after closing position!")
            
        # Check trading logger
        if env.trading_logger:
            print("\n✓ Trading logger is active")
            report = env.trading_logger.generate_trading_report()
            print(f"  - Total errors logged: {report['total_errors']}")
            print(f"  - Trades with None prices: {report['trades_with_none_prices']}")
        
    except Exception as e:
        print(f"✗ Error in trading environment: {e}")
        import traceback
        traceback.print_exc()

def check_log_files():
    """Check trading log files for errors"""
    print("\n=== LOG FILE CHECK ===")
    
    log_dirs = ['logs/trading', 'logs/futures_env', 'logs']
    
    for log_dir in log_dirs:
        if Path(log_dir).exists():
            log_files = list(Path(log_dir).glob('*.log'))
            if log_files:
                print(f"\nFound {len(log_files)} log files in {log_dir}:")
                
                # Check most recent error log
                error_logs = [f for f in log_files if 'error' in f.name.lower()]
                if error_logs:
                    recent_error_log = max(error_logs, key=lambda f: f.stat().st_mtime)
                    print(f"\nMost recent error log: {recent_error_log.name}")
                    
                    # Read last few lines
                    with open(recent_error_log, 'r') as f:
                        lines = f.readlines()
                        if lines:
                            print("Last 5 error entries:")
                            for line in lines[-5:]:
                                print(f"  {line.strip()}")

def main():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("TRADING SYSTEM DIAGNOSTIC REPORT")
    print(f"Generated: {datetime.datetime.now()}")
    print("=" * 60)
    
    # Check data loading
    df = check_data_loading()
    
    if df is not None:
        # Check TimeSeriesState creation
        states = check_timeseries_states(df)
        
        if states:
            # Check trading environment
            check_trading_environment(states)
    
    # Check log files
    check_log_files()
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()