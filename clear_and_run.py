#!/usr/bin/env python3
"""
Clear any stuck sessions and run training
"""
import os
import sys
import psycopg2
from urllib.parse import urlparse

# Get database URL from environment
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')

def clear_active_sessions():
    """Clear any stuck active sessions"""
    try:
        # Parse database URL
        result = urlparse(DATABASE_URL)
        
        # Connect to database
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
        
        with conn.cursor() as cur:
            # Update any active sessions to completed
            cur.execute("""
                UPDATE trading_sessions 
                SET is_active = false, 
                    end_time = CURRENT_TIMESTAMP 
                WHERE is_active = true
            """)
            
            rows_updated = cur.rowcount
            conn.commit()
            
            if rows_updated > 0:
                print(f"✓ Cleared {rows_updated} stuck session(s)")
            else:
                print("✓ No stuck sessions found")
                
        conn.close()
        
    except Exception as e:
        print(f"Warning: Could not clear sessions: {e}")
        print("Continuing anyway...")

if __name__ == "__main__":
    print("Clearing any stuck sessions...")
    clear_active_sessions()
    
    print("\nStarting training...")
    os.system("python train_local.py")