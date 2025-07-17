#!/usr/bin/env python3
"""Generate large test data file for training"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Generate 2.5 million rows (enough for 10,000+ episodes)
print("Generating 2.5 million rows of NQ futures data...")
num_rows = 2_500_000

# Start from a realistic date
start_date = datetime(2020, 1, 2, 9, 30, 0)
timestamps = [start_date + timedelta(minutes=i) for i in range(num_rows)]

# Generate realistic NQ futures prices with proper volatility
base_price = 14880
daily_volatility = 0.015  # 1.5% daily volatility
minute_volatility = daily_volatility / np.sqrt(390)  # 390 minutes in trading day

# Generate price series with random walk
returns = np.random.normal(0, minute_volatility, num_rows)
price_series = base_price * np.exp(np.cumsum(returns))

# Add some trend
trend = np.linspace(0, 1000, num_rows)  # Gradual uptrend
price_series += trend

# Generate OHLCV data
high_prices = price_series + np.abs(np.random.normal(0, 5, num_rows))
low_prices = price_series - np.abs(np.random.normal(0, 5, num_rows))
open_prices = price_series + np.random.normal(0, 2, num_rows)
close_prices = price_series + np.random.normal(0, 2, num_rows)
volumes = np.random.randint(100, 10000, num_rows)

# Create DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
})

# Add technical indicators
print("Adding technical indicators...")

# Simple moving averages
data['sma_20'] = data['close'].rolling(20).mean()
data['sma_50'] = data['close'].rolling(50).mean()

# Price changes
data['returns'] = data['close'].pct_change()

# Volume indicators
data['volume_sma'] = data['volume'].rolling(20).mean()

# Add more random indicators to match expected feature count
for i in range(50):  # Add 50 more features to get ~60 total
    data[f'indicator_{i}'] = np.random.randn(num_rows)

# Fill NaN values from rolling calculations
data = data.fillna(0)

# Save to file
filename = 'large_test_data.csv'
print(f"Saving to {filename}...")
data.to_csv(filename, index=False)

file_size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
print(f"✓ Generated {num_rows:,} rows ({file_size_mb:.1f} MB in memory)")
print(f"✓ Saved to {filename}")
print(f"✓ Features: {len(data.columns)} columns")
print(f"✓ Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
print("\nYou can now run:")
print(f"python train_standalone.py --num-gpus 4 --episodes 10000 --data-file {filename}")