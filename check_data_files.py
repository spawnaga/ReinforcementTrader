#!/usr/bin/env python3
"""
Check if data files are accessible
"""
import os
import pandas as pd

def check_data():
    print("Checking data files...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"✗ Data directory '{data_dir}' not found")
        return
    
    print(f"✓ Data directory exists: {data_dir}")
    
    # List all files in data directory
    files = os.listdir(data_dir)
    print(f"\nFiles in data directory:")
    for f in files:
        filepath = os.path.join(data_dir, f)
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"  - {f} ({size:.2f} MB)")
    
    # Try to load NQ data
    nq_files = [f for f in files if 'NQ' in f.upper() and (f.endswith('.csv') or f.endswith('.txt'))]
    
    if nq_files:
        print(f"\nFound {len(nq_files)} NQ data files")
        for nq_file in nq_files:
            filepath = os.path.join(data_dir, nq_file)
            try:
                # Try to read first few rows
                df = pd.read_csv(filepath, nrows=5)
                print(f"\n✓ Successfully read {nq_file}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Shape: {df.shape}")
            except Exception as e:
                print(f"\n✗ Error reading {nq_file}: {str(e)}")
    else:
        print("\n✗ No NQ data files found")
        
    # Check database
    print("\nChecking database...")
    try:
        from extensions import db
        from models import MarketData
        from app import app
        
        with app.app_context():
            count = MarketData.query.filter_by(symbol='NQ').count()
            print(f"✓ Database has {count} NQ records")
    except Exception as e:
        print(f"✗ Database error: {str(e)}")

if __name__ == "__main__":
    check_data()