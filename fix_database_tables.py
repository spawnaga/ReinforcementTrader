#!/usr/bin/env python3
"""
Create missing database tables including algorithm_config
"""

import os
import sys
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_missing_tables():
    """Create all missing database tables"""
    
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
        
        # Create algorithm_config table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS algorithm_config (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL UNIQUE,
                algorithm_type VARCHAR(50) NOT NULL,
                parameters JSON NOT NULL,
                description TEXT,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✓ Created algorithm_config table")
        
        # Insert default algorithm configurations
        cursor.execute("""
            INSERT INTO algorithm_config (name, algorithm_type, parameters, description)
            VALUES 
            ('ANE-PPO Default', 'ANE_PPO', '{"learning_rate": 0.0003, "gamma": 0.99, "transformer_layers": 2, "attention_dim": 256}', 'Default ANE-PPO configuration'),
            ('DQN Default', 'DQN', '{"learning_rate": 0.001, "gamma": 0.99, "epsilon": 1.0, "epsilon_decay": 0.995}', 'Default DQN configuration'),
            ('Genetic Default', 'GENETIC', '{"population_size": 50, "generations": 100, "mutation_rate": 0.1}', 'Default Genetic Algorithm configuration')
            ON CONFLICT (name) DO NOTHING;
        """)
        print("✓ Inserted default algorithm configurations")
        
        # Check if cont_fut table exists, create if not
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cont_fut (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL UNIQUE,
                name VARCHAR(100) NOT NULL,
                exchange VARCHAR(20) NOT NULL,
                tick_size FLOAT NOT NULL,
                value_per_tick FLOAT NOT NULL,
                margin FLOAT,
                trading_hours VARCHAR(100),
                expiry_pattern VARCHAR(50),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("✓ Created cont_fut table")
        
        # Insert default futures contracts
        cursor.execute("""
            INSERT INTO cont_fut (symbol, name, exchange, tick_size, value_per_tick)
            VALUES 
            ('NQ', 'E-mini Nasdaq-100', 'CME', 0.25, 5.0),
            ('ES', 'E-mini S&P 500', 'CME', 0.25, 12.50)
            ON CONFLICT (symbol) DO NOTHING;
        """)
        print("✓ Inserted default futures contracts")
        
        # Fix market_data table if needed
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='market_data' AND column_name='symbol';
        """)
        
        if cursor.fetchone() is None:
            cursor.execute("""
                ALTER TABLE market_data 
                ADD COLUMN symbol VARCHAR(10) DEFAULT 'NQ' NOT NULL;
            """)
            print("✓ Added 'symbol' column to market_data")
            
        # Show all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        print("\nAll tables in database:")
        print("-" * 30)
        for row in cursor.fetchall():
            cursor.execute(f"SELECT COUNT(*) FROM {row[0]}")
            count = cursor.fetchone()[0]
            print(f"{row[0]:25} ({count} records)")
            
        cursor.close()
        conn.close()
        
        print("\n✓ Database tables fixed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("Creating missing database tables...")
    print("="*50)
    
    if create_missing_tables():
        print("\nDatabase is now ready for use.")
    else:
        print("\nFailed to create tables. Please check the error messages above.")