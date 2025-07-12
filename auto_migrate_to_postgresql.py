#!/usr/bin/env python3
"""
Automatic migration from SQLite to PostgreSQL
"""

import os
import logging
from sqlalchemy import create_engine, text
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_data():
    """Migrate data from SQLite to PostgreSQL"""
    
    sqlite_url = 'sqlite:///instance/trading_system.db'
    postgres_url = os.environ.get('DATABASE_URL')
    
    if not postgres_url:
        logger.error("PostgreSQL DATABASE_URL not found")
        return False
    
    logger.info("Starting automatic migration to PostgreSQL...")
    
    try:
        sqlite_engine = create_engine(sqlite_url)
        postgres_engine = create_engine(postgres_url)
        
        # Get tables from SQLite
        with sqlite_engine.connect() as conn:
            tables_query = text("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            tables = [row[0] for row in conn.execute(tables_query)]
        
        logger.info(f"Found {len(tables)} tables to migrate")
        
        # Migrate each table
        for table_name in tables:
            try:
                df = pd.read_sql_table(table_name, sqlite_engine)
                
                if len(df) > 0:
                    df.to_sql(table_name, postgres_engine, if_exists='replace', index=False)
                    logger.info(f"✓ Migrated {len(df)} rows from '{table_name}'")
                else:
                    logger.info(f"✓ Table '{table_name}' is empty")
                    
            except Exception as e:
                logger.error(f"Error migrating '{table_name}': {e}")
        
        logger.info("✓ Migration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False

if __name__ == '__main__':
    migrate_data()