#!/usr/bin/env python3
"""
Prepare your NQ futures data for training
This script handles the full workflow from raw data to training-ready format
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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_data_loader import GPUDataLoader
from technical_indicators import TechnicalIndicators

def prepare_data_for_training(filepath=None):
    """
    Complete workflow to prepare data for training
    
    Args:
        filepath: Path to your data file (if not in standard location)
    """
    
    # Initialize GPU data loader directly (avoids database dependency)
    gpu_loader = GPUDataLoader()
    
    # Step 1: Load data
    logger.info("Step 1: Loading NQ futures data...")
    
    if filepath:
        df = gpu_loader.load_nq_data(filepath)
    else:
        # Look for NQ files in data directory
        data_files = list(Path('./data').glob('NQ*.txt')) + list(Path('./data').glob('NQ*.csv'))
        if not data_files:
            logger.error("No NQ data files found in ./data directory")
            return None, None
        
        # Use the largest file
        largest_file = max(data_files, key=lambda f: f.stat().st_size)
        logger.info(f"Using data file: {largest_file}")
        df = gpu_loader.load_nq_data(str(largest_file))
    
    if df is None or len(df) == 0:
        logger.error("Failed to load data. Please check your data file.")
        return None, None
    
    logger.info(f"Loaded {len(df):,} rows of data")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Step 2: Add technical indicators
    logger.info("\nStep 2: Adding technical indicators...")
    
    # Initialize technical indicators with dataframe
    ti = TechnicalIndicators(df)
    
    # Add all indicators
    df = ti.add_all_indicators()
    
    logger.info(f"Added {len(df.columns) - 5} technical indicators")
    logger.info(f"Total features: {len(df.columns)}")
    
    # Step 3: Clean data
    logger.info("\nStep 3: Cleaning data...")
    
    # Remove rows with NaN values (from indicator calculations)
    before_clean = len(df)
    df = df.dropna()
    after_clean = len(df)
    
    logger.info(f"Removed {before_clean - after_clean} rows with NaN values")
    logger.info(f"Clean data: {len(df):,} rows")
    
    # Step 4: Create train/test split
    logger.info("\nStep 4: Creating train/test split...")
    
    # Use 80% for training, 20% for testing
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
    
    # Save files
    train_file = processed_dir / 'NQ_train_processed.parquet'
    test_file = processed_dir / 'NQ_test_processed.parquet'
    
    train_df.to_parquet(train_file, compression='snappy')
    test_df.to_parquet(test_file, compression='snappy')
    
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

def create_training_states(df, sequence_length=60, max_states=None):
    """
    Create TimeSeriesState objects for training
    
    Args:
        df: DataFrame with OHLCV and indicators
        sequence_length: Length of each sequence
        max_states: Maximum number of states to create (None for all)
    
    Returns:
        List of TimeSeriesState objects
    """
    from gym_futures.envs.utils import TimeSeriesState
    
    logger.info(f"\nCreating training states (sequence_length={sequence_length})...")
    
    states = []
    total_possible = len(df) - sequence_length + 1
    
    if max_states:
        # Create evenly spaced indices
        indices = np.linspace(0, total_possible - 1, max_states, dtype=int)
    else:
        indices = range(total_possible)
    
    for i, idx in enumerate(indices):
        if i % 10000 == 0:
            logger.info(f"Created {i:,}/{len(indices):,} states...")
        
        state_data = df.iloc[idx:idx+sequence_length]
        state = TimeSeriesState(
            data=state_data,
            current_index=sequence_length - 1
        )
        states.append(state)
    
    logger.info(f"Created {len(states):,} training states")
    return states

def main():
    parser = argparse.ArgumentParser(description='Prepare NQ futures data for training')
    parser.add_argument('--file', type=str, help='Path to your data file')
    parser.add_argument('--max-states', type=int, help='Maximum training states to create')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length for states')
    parser.add_argument('--test-training', action='store_true', help='Run a quick training test')
    args = parser.parse_args()
    
    # Prepare data
    train_df, test_df = prepare_data_for_training(args.file)
    
    if train_df is None:
        return
    
    # Create training states if requested
    if args.test_training:
        logger.info("\n" + "="*60)
        logger.info("TESTING TRAINING SETUP")
        logger.info("="*60)
        
        # Create sample states
        sample_size = min(1000, len(train_df) - args.sequence_length)
        train_states = create_training_states(train_df, args.sequence_length, sample_size)
        
        # Create states for testing
        logger.info("Creating training states for environment test...")
        
        # Test with small episode count
        logger.info("\nCreating sample environment for testing...")
        
        try:
            # Import here to avoid circular imports
            from gym_futures.envs.futures_env import FuturesEnv
            
            # Create a simple test environment
            env = FuturesEnv(
                states=train_states[:10],
                value_per_tick=5.0,
                tick_size=0.25,
                execution_cost_per_order=5.0,
                session_id=1
            )
            
            # Test environment reset
            state = env.reset()
            logger.info(f"✓ Environment created successfully")
            logger.info(f"✓ State shape: {state.shape if hasattr(state, 'shape') else 'TimeSeriesState'}")
            logger.info(f"✓ Your data is ready for full training!")
            
        except Exception as e:
            logger.error(f"Environment test error: {str(e)}")
            logger.info("Data preparation completed successfully, but environment test failed")
            logger.info("You can still use the prepared data files for training")

if __name__ == '__main__':
    main()