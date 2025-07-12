#!/usr/bin/env python3
"""
Script to help load your actual 8M row NQ futures data
"""
import os
import pandas as pd
from sqlalchemy import text
from extensions import db
from models import MarketData
from app import app

def main():
    print("\n🔍 Looking for your NQ futures data file...")
    print("=" * 60)
    
    # Common locations where the data might be
    locations = [
        # Current directory
        ".",
        "./data",
        "./data_cache",
        # Home directory locations
        os.path.expanduser("~/PycharmProjects/ReinforcementTrader/data"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Documents"),
        os.path.expanduser("~/Desktop"),
        # Common data locations
        "/mnt/data",
        "/data",
    ]
    
    large_files = []
    
    for location in locations:
        if os.path.exists(location):
            try:
                for root, dirs, files in os.walk(location):
                    for file in files:
                        if 'NQ' in file.upper() and (file.endswith('.csv') or file.endswith('.txt')):
                            filepath = os.path.join(root, file)
                            try:
                                size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                                if size > 10:  # Only files larger than 10MB
                                    large_files.append((filepath, size))
                            except:
                                pass
            except:
                pass
    
    if not large_files:
        print("\n❌ No large NQ data files found!")
        print("\nPlease specify the path to your 8M row NQ futures data file:")
        filepath = input("File path: ").strip()
        
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            large_files.append((filepath, size))
        else:
            print(f"❌ File not found: {filepath}")
            return
    
    print(f"\n✅ Found {len(large_files)} large NQ data file(s):")
    for i, (path, size) in enumerate(large_files):
        print(f"  {i+1}. {path} ({size:.2f} MB)")
    
    if len(large_files) == 1:
        filepath, size = large_files[0]
        print(f"\n📂 Using: {filepath}")
    else:
        choice = input(f"\nSelect file (1-{len(large_files)}): ")
        filepath, size = large_files[int(choice)-1]
    
    print(f"\n📊 Loading data from {filepath}...")
    
    with app.app_context():
        try:
            # Load data with progress updates
            chunk_size = 100000
            chunks = []
            total_rows = 0
            
            # First pass to count rows and check format
            for i, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
                total_rows += len(chunk)
                if i == 0:
                    print(f"  Columns: {list(chunk.columns)}")
                    print(f"  First row: {chunk.iloc[0].to_dict()}")
                if i % 10 == 0:
                    print(f"  Counted {total_rows:,} rows so far...")
            
            print(f"\n✅ Total rows in file: {total_rows:,}")
            
            # Clear existing data
            print("\n🗑️  Clearing existing NQ data from database...")
            with db.engine.connect() as conn:
                conn.execute(text("DELETE FROM market_data WHERE symbol = 'NQ'"))
                conn.commit()
            
            # Load data in batches
            print(f"\n📥 Loading data to database...")
            loaded = 0
            
            for chunk_num, chunk in enumerate(pd.read_csv(filepath, chunksize=chunk_size)):
                # Standardize columns
                chunk.columns = chunk.columns.str.lower()
                
                # Convert to database records
                records = []
                for _, row in chunk.iterrows():
                    record = MarketData(
                        symbol='NQ',
                        timestamp=pd.to_datetime(row['timestamp']),
                        open_price=float(row['open']),
                        high_price=float(row['high']),
                        low_price=float(row['low']),
                        close_price=float(row['close']),
                        volume=int(row['volume'])
                    )
                    records.append(record)
                
                # Bulk insert
                db.session.bulk_save_objects(records)
                db.session.commit()
                
                loaded += len(records)
                progress = (loaded / total_rows) * 100
                print(f"  Progress: {loaded:,}/{total_rows:,} rows ({progress:.1f}%)")
            
            print(f"\n✅ Successfully loaded {loaded:,} rows to database!")
            
            # Verify
            with db.engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM market_data WHERE symbol = 'NQ'")).scalar()
                print(f"✅ Database now contains {count:,} NQ records")
            
            print("\n🎉 Data loading complete! You can now start training.")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    main()