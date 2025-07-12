#!/usr/bin/env python3
"""
Migrate data from local CSV files to PostgreSQL database
"""
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from extensions import db
from models import MarketData
from app import app

def migrate_data():
    """Migrate all NQ data from CSV files to database"""
    with app.app_context():
        data_dir = "data"
        
        # Find the user's actual data file
        # Based on the attached logs, it seems the user has a large file somewhere
        print("Looking for data files...")
        
        # First, let's check what files are available
        for root, dirs, files in os.walk("."):
            for file in files:
                if 'NQ' in file.upper() and (file.endswith('.csv') or file.endswith('.txt')):
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    if size > 1:  # Only interested in files larger than 1MB
                        print(f"Found potential data file: {filepath} ({size:.2f} MB)")
                        
                        try:
                            # Try to load and check the file
                            df = pd.read_csv(filepath, nrows=10)
                            print(f"  Columns: {list(df.columns)}")
                            print(f"  Sample shape: {df.shape}")
                            
                            # Ask user confirmation
                            response = input(f"\nLoad this file to database? (y/n): ")
                            if response.lower() == 'y':
                                load_file_to_db(filepath)
                                
                        except Exception as e:
                            print(f"  Error reading: {str(e)}")
                            
        print("\nMigration complete!")
        
def load_file_to_db(filepath):
    """Load a specific file to database"""
    print(f"\nLoading {filepath}...")
    
    try:
        # Read the file
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows")
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Ensure we have the required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            print("Missing required columns. Attempting to fix...")
            # Handle different column naming conventions
            rename_map = {
                'date': 'timestamp',
                'datetime': 'timestamp',
                'time': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vol': 'volume'
            }
            df.rename(columns=rename_map, inplace=True)
            
        # Clear existing data
        print("Clearing existing NQ data...")
        with db.engine.connect() as conn:
            conn.execute(text("DELETE FROM market_data WHERE symbol = 'NQ'"))
            conn.commit()
            
        # Insert data in batches
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            records = []
            
            for _, row in batch.iterrows():
                record = MarketData(
                    symbol='NQ',
                    timestamp=pd.to_datetime(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                records.append(record)
                
            db.session.bulk_save_objects(records)
            db.session.commit()
            total_inserted += len(records)
            
            if total_inserted % 10000 == 0:
                print(f"  Inserted {total_inserted} records...")
                
        print(f"Successfully loaded {total_inserted} records to database!")
        
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        db.session.rollback()

if __name__ == "__main__":
    migrate_data()