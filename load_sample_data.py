#!/usr/bin/env python3
"""
Load sample market data into the PostgreSQL database for testing
"""
import pandas as pd
from datetime import datetime
from extensions import db
from models import MarketData
from app import app
import os
from sqlalchemy import text

def load_sample_data():
    """Load sample NQ data into the PostgreSQL database"""
    # Check if data file exists
    data_file = "data/NQ_test_data.txt"
    if not os.path.exists(data_file):
        # Try alternative paths for local development
        alt_paths = [
            "/home/alex/PycharmProjects/ReinforcementTrader/data/NQ_test_data.txt",
            "~/PycharmProjects/ReinforcementTrader/data/NQ_test_data.txt"
        ]
        for alt_path in alt_paths:
            expanded_path = os.path.expanduser(alt_path)
            if os.path.exists(expanded_path):
                data_file = expanded_path
                break
        else:
            print(f"Error: Could not find NQ_test_data.txt in any of the expected locations")
            print("Please ensure you have data files in the 'data' directory")
            return False
    
    print(f"Loading data from {data_file}...")
    
    # Read the data file
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows from file")
    
    # Check columns
    print(f"Columns in file: {list(df.columns)}")
    
    with app.app_context():
        # Clear existing market data
        existing_count = MarketData.query.count()
        if existing_count > 0:
            print(f"Clearing {existing_count} existing market data records...")
            MarketData.query.delete()
            db.session.commit()
        
        # Convert and insert data
        records_added = 0
        for idx, row in df.iterrows():
            try:
                # Parse timestamp
                if 'timestamp' in row:
                    timestamp = pd.to_datetime(row['timestamp'])
                elif 'date' in row:
                    timestamp = pd.to_datetime(row['date'])
                else:
                    timestamp = datetime.now()
                
                # Create market data record
                market_data = MarketData(
                    timestamp=timestamp,
                    symbol='NQ',
                    open_price=float(row['open']),
                    high_price=float(row['high']),
                    low_price=float(row['low']),
                    close_price=float(row['close']),
                    volume=int(row['volume']) if 'volume' in row else 1000,
                    timeframe='1min'
                )
                
                db.session.add(market_data)
                records_added += 1
                
                # Commit in batches
                if records_added % 100 == 0:
                    db.session.commit()
                    print(f"Added {records_added} records...")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                continue
        
        # Final commit
        db.session.commit()
        print(f"\nSuccessfully loaded {records_added} market data records")
        
        # Verify data was loaded
        final_count = MarketData.query.count()
        print(f"Total records in database: {final_count}")
        
        # Show sample data
        sample = MarketData.query.order_by(MarketData.timestamp.desc()).limit(5).all()
        print("\nSample data (last 5 records):")
        for record in sample:
            print(f"  {record.timestamp}: O={record.open_price}, H={record.high_price}, "
                  f"L={record.low_price}, C={record.close_price}, V={record.volume}")
        
        return True

if __name__ == "__main__":
    # Check PostgreSQL connection
    print("Checking PostgreSQL connection...")
    print(f"DATABASE_URL: {os.environ.get('DATABASE_URL', 'Not set')}")
    
    try:
        with app.app_context():
            # Test connection
            result = db.session.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to PostgreSQL: {version}")
            
            # Check if market_data table exists
            result = db.session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'market_data'
                )
            """))
            table_exists = result.scalar()
            
            if not table_exists:
                print("❌ market_data table does not exist. Creating tables...")
                db.create_all()
                print("✅ Tables created successfully")
    except Exception as e:
        print(f"❌ Database connection error: {str(e)}")
        print("\nPlease ensure:")
        print("1. PostgreSQL is running")
        print("2. DATABASE_URL is set correctly in your environment or .env file")
        print("3. The database 'reinforcement_trader' exists")
        exit(1)
    
    if load_sample_data():
        print("\n✅ Sample data loaded successfully!")
        print("You can now start training with the API.")
    else:
        print("\n❌ Failed to load sample data")