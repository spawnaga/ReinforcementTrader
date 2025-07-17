#!/usr/bin/env python3
"""Create test data for debugging the 11735 reward bug"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create test data with realistic NQ futures prices
np.random.seed(42)
n_rows = 1000  # Enough for a few episodes

# Generate timestamps
start_date = datetime(2008, 1, 2, 7, 0)
timestamps = [start_date + timedelta(hours=i) for i in range(n_rows)]

# Generate realistic price movements
base_price = 3350.0
prices = [base_price]
for i in range(1, n_rows):
    # Random walk with mean reversion
    change = np.random.normal(0, 5) - 0.01 * (prices[-1] - base_price)
    new_price = prices[-1] + change
    prices.append(new_price)

# Create OHLCV data
data = []
for i, (ts, close) in enumerate(zip(timestamps, prices)):
    # Generate realistic OHLC from close price
    volatility = np.random.uniform(0.1, 0.3)
    high = close + np.random.uniform(0, 10) * volatility
    low = close - np.random.uniform(0, 10) * volatility
    open_price = close + np.random.uniform(-5, 5) * volatility
    volume = np.random.randint(1000, 5000)
    
    data.append({
        'timestamp': ts,
        'open': round(open_price, 2),
        'high': round(high, 2),
        'low': round(low, 2),
        'close': round(close, 2),
        'volume': volume
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('./data/processed/NQ_train_processed.csv', index=False)
print(f"Created test data with {len(df)} rows")
print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
print("Sample data:")
print(df.head())