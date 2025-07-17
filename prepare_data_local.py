#!/usr/bin/env python3
"""
Local version of prepare_training_data.py that works without database
Run this in your WSL2 environment
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nq_data_simple(filepath):
    """Simple data loader without GPU acceleration"""
    logger.info(f"Loading data from {filepath}")
    
    try:
        # Read CSV with no headers
        df = pd.read_csv(
            filepath,
            header=None,
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def add_technical_indicators(df):
    """Add basic technical indicators"""
    logger.info("Adding technical indicators...")
    
    # Simple Moving Averages
    for period in [5, 10, 20, 50, 100]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
    
    # Exponential Moving Averages
    for period in [5, 10, 20, 50]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # RSI
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = calculate_rsi(df['close'])
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # MACD
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price features
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Time features - raw values
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek
    
    # Cyclical time encoding using sin/cos transformations
    # Convert time to minutes since midnight
    minutes_since_midnight = df['hour'] * 60 + df['minute']
    
    # Time of day encoding (24-hour cycle)
    MINUTES_IN_DAY = 24 * 60  # 1440 minutes
    df['sin_time'] = np.sin(2 * np.pi * minutes_since_midnight / MINUTES_IN_DAY)
    df['cos_time'] = np.cos(2 * np.pi * minutes_since_midnight / MINUTES_IN_DAY)
    
    # Day of week encoding (7-day cycle)
    DAYS_IN_WEEK = 7
    df['sin_weekday'] = np.sin(2 * np.pi * df['day_of_week'] / DAYS_IN_WEEK)
    df['cos_weekday'] = np.cos(2 * np.pi * df['day_of_week'] / DAYS_IN_WEEK)
    
    # Hour of day encoding (24-hour cycle) - more granular than time
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df

def prepare_data_for_training(filepath):
    """Main data preparation workflow"""
    
    # Step 1: Load data
    logger.info("Step 1: Loading NQ futures data...")
    df = load_nq_data_simple(filepath)
    
    logger.info(f"Loaded {len(df):,} rows of data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Add technical indicators
    logger.info("\nStep 2: Adding technical indicators...")
    df = add_technical_indicators(df)
    
    logger.info(f"Added {len(df.columns) - 5} technical indicators")
    logger.info(f"Total features: {len(df.columns)}")
    
    # Log cyclical features specifically
    cyclical_features = ['sin_time', 'cos_time', 'sin_weekday', 'cos_weekday', 'sin_hour', 'cos_hour']
    logger.info(f"Including {len(cyclical_features)} cyclical time features: {', '.join(cyclical_features)}")
    
    # Step 3: Clean data
    logger.info("\nStep 3: Cleaning data...")
    before_clean = len(df)
    df = df.dropna()
    after_clean = len(df)
    
    logger.info(f"Removed {before_clean - after_clean} rows with NaN values")
    logger.info(f"Clean data: {len(df):,} rows")
    
    # Step 4: Create train/test split
    logger.info("\nStep 4: Creating train/test split...")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    logger.info(f"Training data: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Test data: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Step 5: Save processed data
    logger.info("\nStep 5: Saving processed data...")
    
    # Create processed data directory
    processed_dir = Path('./data/processed')
    processed_dir.mkdir(exist_ok=True)
    
    # Save as CSV for compatibility
    train_file = processed_dir / 'NQ_train_processed.csv'
    test_file = processed_dir / 'NQ_test_processed.csv'
    
    train_df.to_csv(train_file)
    test_df.to_csv(test_file)
    
    logger.info(f"Saved training data to {train_file}")
    logger.info(f"Saved test data to {test_file}")
    
    # Step 6: Display summary
    logger.info("\n" + "="*60)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total rows processed: {len(df):,}")
    logger.info(f"Features created: {len(df.columns)}")
    logger.info(f"Training samples: {len(train_df):,}")
    logger.info(f"Test samples: {len(test_df):,}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Show feature list
    logger.info("\nFeatures available for training:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")
    
    return train_df, test_df

def main():
    parser = argparse.ArgumentParser(description='Prepare NQ futures data for training (local version)')
    parser.add_argument('--file', type=str, required=True, help='Path to your data file')
    parser.add_argument('--sample', type=int, help='Sample size to process (for testing)')
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return
    
    # Prepare data
    train_df, test_df = prepare_data_for_training(args.file)
    
    logger.info("\nNext steps:")
    logger.info("1. The processed data files are saved in ./data/processed/")
    logger.info("2. You can use these files for training with your preferred ML framework")
    logger.info("3. To use with the trading system, copy these files to your Replit environment")

if __name__ == '__main__':
    main()