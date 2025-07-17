#!/usr/bin/env python3
"""
Fix the market_data table schema by adding missing columns
"""

import os
import psycopg2
from psycopg2 import sql
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fix_market_data_table():
    """Add missing columns to market_data table"""
    
    # Parse database URL
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        return False
        
    result = urlparse(database_url)
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("Connected to database successfully")
        
        # Check if symbol column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='market_data' AND column_name='symbol';
        """)
        
        if cursor.fetchone() is None:
            print("Adding 'symbol' column to market_data table...")
            cursor.execute("""
                ALTER TABLE market_data 
                ADD COLUMN symbol VARCHAR(10) DEFAULT 'NQ' NOT NULL;
            """)
            print("✓ Added 'symbol' column")
        else:
            print("✓ 'symbol' column already exists")
            
        # Check if timeframe column exists
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='market_data' AND column_name='timeframe';
        """)
        
        if cursor.fetchone() is None:
            print("Adding 'timeframe' column to market_data table...")
            cursor.execute("""
                ALTER TABLE market_data 
                ADD COLUMN timeframe VARCHAR(10) DEFAULT '1min';
            """)
            print("✓ Added 'timeframe' column")
        else:
            print("✓ 'timeframe' column already exists")
            
        # Update existing records to have NQ symbol if needed
        cursor.execute("""
            UPDATE market_data 
            SET symbol = 'NQ' 
            WHERE symbol IS NULL OR symbol = '';
        """)
        
        # Create index on symbol and timeframe for better query performance
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe 
                ON market_data(symbol, timeframe);
            """)
            print("✓ Created index on symbol and timeframe")
        except Exception as e:
            print(f"Note: Index might already exist: {e}")
            
        # Show current table structure
        cursor.execute("""
            SELECT column_name, data_type, is_nullable 
            FROM information_schema.columns 
            WHERE table_name = 'market_data' 
            ORDER BY ordinal_position;
        """)
        
        print("\nCurrent market_data table structure:")
        print("-" * 50)
        for row in cursor.fetchall():
            print(f"{row[0]:20} {row[1]:20} {'NULL' if row[2] == 'YES' else 'NOT NULL'}")
            
        # Count records
        cursor.execute("SELECT COUNT(*) FROM market_data;")
        count = cursor.fetchone()[0]
        print(f"\nTotal records in market_data: {count}")
        
        cursor.close()
        conn.close()
        
        print("\n✓ Database schema fixed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("Fixing market_data table schema...")
    print("="*50)
    
    if fix_market_data_table():
        print("\nYou can now access the /api/market_data endpoint without errors.")
    else:
        print("\nFailed to fix database schema. Please check the error messages above.")