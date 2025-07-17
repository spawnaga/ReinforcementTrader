#!/usr/bin/env python3
"""
Setup database tables for the trading system
"""
import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

def setup_database():
    """Create all necessary tables"""
    try:
        result = urlparse(DATABASE_URL)
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
        
        with conn.cursor() as cur:
            # Create trading_sessions table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trading_sessions (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    algorithm VARCHAR(50) NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE,
                    total_episodes INTEGER DEFAULT 0,
                    completed_episodes INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create trades table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES trading_sessions(id),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ticker VARCHAR(10) NOT NULL,
                    action VARCHAR(10) NOT NULL,
                    price DECIMAL(10, 2),
                    quantity INTEGER,
                    profit_loss DECIMAL(10, 2),
                    position INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create market_data table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    open_price DECIMAL(10, 2),
                    high_price DECIMAL(10, 2),
                    low_price DECIMAL(10, 2),
                    close_price DECIMAL(10, 2),
                    volume BIGINT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, timestamp)
                )
            """)
            
            # Create training_metrics table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES trading_sessions(id),
                    episode INTEGER,
                    reward DECIMAL(10, 4),
                    loss DECIMAL(10, 4),
                    sharpe_ratio DECIMAL(10, 4),
                    max_drawdown DECIMAL(10, 4),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create algorithm_configs table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS algorithm_configs (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES trading_sessions(id),
                    algorithm_name VARCHAR(50),
                    config_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            print("✓ Database tables created successfully!")
            
        conn.close()
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise

if __name__ == "__main__":
    print("Setting up database tables...")
    setup_database()