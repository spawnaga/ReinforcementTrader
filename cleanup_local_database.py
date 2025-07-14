#!/usr/bin/env python3
"""
Clean up your local PostgreSQL database for the AI Trading System
This will remove all training sessions and trades while keeping market data
"""

import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

def load_env_file():
    """Load database URL from .env file"""
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    if key == 'DATABASE_URL':
                        return value
    return None

def cleanup_database():
    """Clean up all training data while preserving market data"""
    
    # Try to get database URL from .env file first
    db_url = load_env_file()
    
    if not db_url:
        # Fall back to environment variable
        db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        print("DATABASE_URL not found in .env file or environment.")
        print("Please enter your PostgreSQL connection string:")
        print("Format: postgresql://username:password@host:port/database")
        db_url = input("> ").strip()
    else:
        print(f"Using database connection from .env file")
    
    try:
        # Connect to database
        print(f"\nConnecting to database...")
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Show current data counts
        print("\n--- BEFORE CLEANUP ---")
        tables = ['market_data', 'trading_session', 'trade', 'training_metrics', 'algorithm_config']
        
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"{table}: {count} records")
            except psycopg2.Error:
                print(f"{table}: table not found")
        
        # Confirm before proceeding
        print("\n⚠️  WARNING: This will delete ALL training sessions and trades!")
        print("Market data will be preserved.")
        response = input("\nProceed with cleanup? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Cleanup cancelled.")
            return
        
        # Perform cleanup
        print("\nCleaning up database...")
        
        # Delete trades first (due to foreign key constraints)
        cur.execute("DELETE FROM trade")
        trades_deleted = cur.rowcount
        print(f"✓ Deleted {trades_deleted} trades")
        
        # Delete training metrics
        cur.execute("DELETE FROM training_metrics")
        metrics_deleted = cur.rowcount
        print(f"✓ Deleted {metrics_deleted} training metrics")
        
        # Delete trading sessions
        cur.execute("DELETE FROM trading_session")
        sessions_deleted = cur.rowcount
        print(f"✓ Deleted {sessions_deleted} trading sessions")
        
        # Delete algorithm configs
        cur.execute("DELETE FROM algorithm_config")
        configs_deleted = cur.rowcount
        print(f"✓ Deleted {configs_deleted} algorithm configs")
        
        # Show final counts
        print("\n--- AFTER CLEANUP ---")
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"{table}: {count} records")
            except psycopg2.Error:
                print(f"{table}: table not found")
        
        # Close connection
        cur.close()
        conn.close()
        
        print("\n✅ Database cleanup complete!")
        print("Your trading system is now in a clean state.")
        print("Market data has been preserved for training.")
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        print("\nPlease check your connection string and try again.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    cleanup_database()