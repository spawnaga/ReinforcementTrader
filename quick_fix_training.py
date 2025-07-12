#!/usr/bin/env python3
"""
Quick fix to get training working with existing database data
"""
import pandas as pd
from sqlalchemy import text
from extensions import db
from models import MarketData
from app import app
import sys

def fix_training():
    """Ensure we have data in the database for training"""
    with app.app_context():
        try:
            # Check current data
            with db.engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM market_data WHERE symbol = 'NQ'")).scalar()
                print(f"Current database has {count} NQ records")
                
                if count == 0:
                    print("\n⚠️  No data in database! Loading sample data...")
                    
                    # Load from CSV files
                    data_files = ['data/NQ_data.csv', 'data/NQ_test_data.txt', 'data/NQ_large_test.csv']
                    all_data = []
                    
                    for file in data_files:
                        try:
                            df = pd.read_csv(file)
                            df.columns = df.columns.str.lower()
                            all_data.append(df)
                            print(f"  Loaded {len(df)} rows from {file}")
                        except Exception as e:
                            print(f"  Could not load {file}: {e}")
                    
                    if all_data:
                        combined = pd.concat(all_data, ignore_index=True)
                        
                        # Insert to database
                        records = []
                        for _, row in combined.iterrows():
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
                        
                        db.session.bulk_save_objects(records)
                        db.session.commit()
                        print(f"\n✅ Loaded {len(records)} records to database!")
                    else:
                        print("\n❌ No data files found!")
                        sys.exit(1)
                
                # Verify final count
                final_count = conn.execute(text("SELECT COUNT(*) FROM market_data WHERE symbol = 'NQ'")).scalar()
                print(f"\n✅ Database now has {final_count} NQ records")
                print("✅ Training should work now!")
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            db.session.rollback()

if __name__ == "__main__":
    fix_training()