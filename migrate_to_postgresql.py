#!/usr/bin/env python3
"""
Migrate data from SQLite to PostgreSQL
This script helps migrate existing data to avoid concurrency issues
"""

import os
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_data():
    """Migrate data from SQLite to PostgreSQL"""
    
    # Get database URLs
    sqlite_url = 'sqlite:///instance/trading_system.db'
    postgres_url = os.environ.get('DATABASE_URL')
    
    if not postgres_url:
        logger.error("PostgreSQL DATABASE_URL not found in environment variables")
        return False
    
    logger.info(f"Migrating from SQLite to PostgreSQL...")
    
    try:
        # Create engines
        sqlite_engine = create_engine(sqlite_url)
        postgres_engine = create_engine(postgres_url)
        
        # Get list of tables from SQLite
        with sqlite_engine.connect() as conn:
            tables_query = text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in conn.execute(tables_query)]
        
        logger.info(f"Found {len(tables)} tables to migrate: {tables}")
        
        # Migrate each table
        for table_name in tables:
            try:
                # Read data from SQLite
                df = pd.read_sql_table(table_name, sqlite_engine)
                
                if len(df) > 0:
                    # Write to PostgreSQL
                    df.to_sql(table_name, postgres_engine, if_exists='append', index=False)
                    logger.info(f"Migrated {len(df)} rows from table '{table_name}'")
                else:
                    logger.info(f"Table '{table_name}' is empty, skipping")
                    
            except Exception as e:
                logger.error(f"Error migrating table '{table_name}': {e}")
                # Continue with other tables
        
        logger.info("Migration completed successfully!")
        
        # Test PostgreSQL connection
        with postgres_engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM market_data"))
            count = result.scalar()
            logger.info(f"PostgreSQL test: Found {count} records in market_data table")
        
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("SQLite to PostgreSQL Migration Tool")
    print("=" * 60)
    
    # Check if PostgreSQL is configured
    if not os.environ.get('DATABASE_URL'):
        print("\nError: PostgreSQL DATABASE_URL not configured")
        print("The system has already provisioned a PostgreSQL database.")
        print("Please ensure the DATABASE_URL environment variable is set.")
        return
    
    print("\nThis will migrate your data from SQLite to PostgreSQL")
    print("PostgreSQL handles concurrent access much better for multi-GPU training")
    
    response = input("\nProceed with migration? (yes/no): ")
    
    if response.lower() == 'yes':
        if migrate_data():
            print("\n✓ Migration completed successfully!")
            print("Your system is now using PostgreSQL for better concurrency")
        else:
            print("\n✗ Migration failed. Check the logs above for details.")
    else:
        print("\nMigration cancelled.")

if __name__ == '__main__':
    main()