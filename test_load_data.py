#!/usr/bin/env python3
"""
Quick test script to load your NQ futures data
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gpu_data_loader import GPUDataLoader

def test_load_data():
    """Test loading your specific data file"""
    
    # Look for .txt files in data folder
    data_files = list(Path('./data').glob('*.txt'))
    
    if not data_files:
        print("No .txt files found in ./data folder")
        print("\nPlease ensure your NQ data file is in the ./data folder")
        print("Expected format: timestamp,open,high,low,close,volume")
        return
    
    # Find the largest file (should be your 302MB file)
    largest_file = max(data_files, key=lambda f: f.stat().st_size)
    file_size_mb = largest_file.stat().st_size / (1024 * 1024)
    
    print(f"\nFound data file: {largest_file}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Initialize GPU data loader
    gpu_loader = GPUDataLoader(chunk_size=100000)
    
    try:
        # Test with a small sample first
        print("\nLoading sample data (first 1000 rows)...")
        df_sample = gpu_loader.load_nq_data(str(largest_file), max_rows=1000)
        
        print(f"\nSuccessfully loaded {len(df_sample)} rows")
        print(f"Columns: {df_sample.columns.tolist()}")
        print(f"Data types:\n{df_sample.dtypes}")
        
        print("\nFirst 5 rows:")
        print(df_sample.head())
        
        print("\nLast 5 rows:")
        print(df_sample.tail())
        
        print("\nData statistics:")
        print(df_sample.describe())
        
        # Ask if user wants to load full data
        print(f"\nFull file contains approximately {file_size_mb * 1000 / 23:.0f}k rows")
        response = input("Load full file? This may take a few minutes (yes/no): ")
        
        if response.lower() == 'yes':
            print("\nLoading full data file...")
            df_full = gpu_loader.load_nq_data(str(largest_file))
            
            print(f"\nSuccessfully loaded {len(df_full):,} rows")
            print(f"Date range: {df_full.index.min()} to {df_full.index.max()}")
            print(f"Memory usage: {df_full.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Save a small sample for quick testing
            sample_file = './data/NQ_sample_1000.csv'
            df_full.head(1000).to_csv(sample_file)
            print(f"\nSaved 1000-row sample to {sample_file} for quick testing")
            
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        print("\nPlease check that your data file has the format:")
        print("timestamp,open,high,low,close,volume")
        print("Example: 2008-01-02 06:00:00,3602.50,3603.75,3601.75,3603.25,184")

if __name__ == '__main__':
    test_load_data()