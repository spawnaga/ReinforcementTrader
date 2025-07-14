#!/usr/bin/env python3
"""
Quick database cleanup script for AI Trading System
Automatically reads credentials from .env file
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database connection from .env file
DATABASE_URL = "postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader"

def cleanup():
    print("🧹 AI Trading System - Database Cleanup")
    print("=" * 50)
    
    try:
        # Connect
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Show current counts
        print("\nCurrent database state:")
        tables = [
            ('market_data', '📊'),
            ('trading_session', '🎯'),
            ('trade', '💰'),
            ('training_metrics', '📈'),
            ('algorithm_config', '⚙️')
        ]
        
        for table, icon in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"{icon} {table}: {count} records")
            except:
                pass
        
        # Confirm
        print("\n⚠️  This will delete ALL training sessions and trades!")
        print("   Market data will be preserved.")
        response = input("\nProceed? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Cancelled.")
            return
        
        # Clean up
        print("\nCleaning...")
        
        cur.execute("DELETE FROM trade")
        print(f"✓ Deleted {cur.rowcount} trades")
        
        cur.execute("DELETE FROM training_metrics")
        print(f"✓ Deleted {cur.rowcount} training metrics")
        
        cur.execute("DELETE FROM trading_session")
        print(f"✓ Deleted {cur.rowcount} sessions")
        
        cur.execute("DELETE FROM algorithm_config")
        print(f"✓ Deleted {cur.rowcount} algorithm configs")
        
        # Final state
        print("\nFinal database state:")
        for table, icon in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                count = cur.fetchone()[0]
                print(f"{icon} {table}: {count} records")
            except:
                pass
        
        cur.close()
        conn.close()
        
        print("\n✅ Database cleanup complete!")
        print("   Your system is ready for fresh training.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure PostgreSQL is running and accessible.")

if __name__ == "__main__":
    cleanup()