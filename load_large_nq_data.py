#!/usr/bin/env python3
"""
Script to load and prepare large NQ futures data for training
Handles 6 million rows efficiently with GPU acceleration if available
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_data_loader import GPUDataLoader
from models import MarketData, db
from app import app

def find_large_txt_file(data_dir='./data'):
    """Find the large .txt data file in the data directory"""
    data_path = Path(data_dir)
    
    # Look for .txt files
    txt_files = list(data_path.glob('*.txt'))
    
    # Find the largest one (should be your 302MB file)
    if txt_files:
        largest_file = max(txt_files, key=lambda f: f.stat().st_size)
        file_size_mb = largest_file.stat().st_size / (1024 * 1024)
        
        if file_size_mb > 100:  # If file is larger than 100MB, it's probably the right one
            logger.info(f"Found large data file: {largest_file} ({file_size_mb:.2f} MB)")
            return str(largest_file)
    
    # If no large .txt file found, ask user
    logger.warning("No large .txt file found in ./data directory")
    return None

def load_nq_data_file(filepath, sample_size=None):
    """
    Load NQ futures data from a headerless CSV/TXT file
    
    Args:
        filepath: Path to the data file
        sample_size: Number of rows to load (None for all)
    
    Returns:
        DataFrame with processed data
    """
    logger.info(f"Loading data from {filepath}")
    
    # Initialize GPU data loader
    gpu_loader = GPUDataLoader(chunk_size=100000)  # 100k rows per chunk for 6M row file
    
    try:
        # Check file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        # For testing, you can use sample_size
        if sample_size:
            logger.info(f"Loading sample of {sample_size:,} rows")
        
        # Load data using GPU loader (handles chunking automatically)
        df = gpu_loader.load_nq_data(filepath, max_rows=sample_size)
        
        logger.info(f"Loaded {len(df):,} rows")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Display sample data
        logger.info("\nFirst 5 rows:")
        print(df.head())
        
        logger.info("\nLast 5 rows:")
        print(df.tail())
        
        # Data statistics
        logger.info("\nData statistics:")
        print(df.describe())
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_to_database(df, symbol='NQ', batch_size=10000):
    """
    Save data to PostgreSQL database in batches
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Symbol name
        batch_size: Number of rows per batch
    """
    logger.info(f"Saving {len(df):,} rows to database...")
    
    with app.app_context():
        try:
            # Delete existing data for this symbol
            existing_count = MarketData.query.filter_by(symbol=symbol).count()
            if existing_count > 0:
                logger.info(f"Removing {existing_count:,} existing rows for {symbol}")
                MarketData.query.filter_by(symbol=symbol).delete()
                db.session.commit()
            
            # Save in batches
            total_rows = len(df)
            saved_rows = 0
            
            for i in range(0, total_rows, batch_size):
                batch_df = df.iloc[i:i+batch_size]
                
                # Create MarketData objects
                market_data_objects = []
                for timestamp, row in batch_df.iterrows():
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=timestamp,
                        open_price=float(row['open']),
                        high_price=float(row['high']),
                        low_price=float(row['low']),
                        close_price=float(row['close']),
                        volume=int(row['volume']),
                        timeframe='1min'
                    )
                    market_data_objects.append(market_data)
                
                # Bulk insert
                db.session.bulk_save_objects(market_data_objects)
                db.session.commit()
                
                saved_rows += len(batch_df)
                logger.info(f"Saved {saved_rows:,}/{total_rows:,} rows ({saved_rows/total_rows*100:.1f}%)")
            
            logger.info(f"Successfully saved all {total_rows:,} rows to database")
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            db.session.rollback()
            raise

def main():
    parser = argparse.ArgumentParser(description='Load large NQ futures data')
    parser.add_argument('--file', type=str, help='Path to data file')
    parser.add_argument('--sample', type=int, help='Number of rows to sample (for testing)')
    parser.add_argument('--save-db', action='store_true', help='Save to database')
    parser.add_argument('--batch-size', type=int, default=10000, help='Database batch size')
    args = parser.parse_args()
    
    # Find or use specified file
    if args.file:
        filepath = args.file
    else:
        filepath = find_large_txt_file()
        if not filepath:
            print("\nPlease specify the path to your NQ data file:")
            print("Example: python load_large_nq_data.py --file ./data/your_nq_data.txt")
            return
    
    # Check if file exists
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return
    
    # Load data
    df = load_nq_data_file(filepath, sample_size=args.sample)
    
    # Save to database if requested
    if args.save_db:
        response = input(f"\nThis will save {len(df):,} rows to the database. Continue? (yes/no): ")
        if response.lower() == 'yes':
            save_to_database(df, batch_size=args.batch_size)
        else:
            logger.info("Database save cancelled")
    else:
        logger.info("\nTo save to database, run with --save-db flag")
        logger.info("For testing with smaller sample: --sample 10000")

if __name__ == '__main__':
    main()