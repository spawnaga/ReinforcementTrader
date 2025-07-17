"""
Setup PostgreSQL database schemas for training tracking
"""
import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def setup_database():
    """Create all necessary tables for training tracking"""
    # Get database URL from environment
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Connect and create schemas
    conn = psycopg2.connect(database_url)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    with open('database/schemas.sql', 'r') as f:
        schema_sql = f.read()
    
    with conn.cursor() as cur:
        try:
            cur.execute(schema_sql)
            print("✓ Training database schemas created successfully!")
        except Exception as e:
            print(f"Error creating schemas: {e}")
            raise
    
    conn.close()

if __name__ == "__main__":
    setup_database()