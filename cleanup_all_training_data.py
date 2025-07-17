#!/usr/bin/env python3
"""
Complete cleanup of all training data, models, and sessions
"""

import os
import shutil
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

def load_database_url():
    """Load DATABASE_URL from environment or .env file"""
    db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        env_path = Path('.env')
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key == 'DATABASE_URL':
                            db_url = value
                            break
    
    return db_url

def cleanup_files():
    """Clean up all model files, logs, and cache"""
    print("🧹 Cleaning up files...")
    
    # Clean models directory
    models_dir = Path('models')
    if models_dir.exists():
        count = 0
        for item in models_dir.glob('**/*.pt'):
            item.unlink()
            count += 1
        for item in models_dir.glob('**/*.pth'):
            item.unlink()
            count += 1
        for item in models_dir.glob('**/*.pkl'):
            item.unlink()
            count += 1
        
        # Remove empty directories
        for dir_path in sorted(models_dir.glob('**/'), reverse=True):
            if dir_path.is_dir() and not list(dir_path.iterdir()):
                dir_path.rmdir()
        
        print(f"✓ Deleted {count} model files")
    
    # Clean logs directory
    logs_dir = Path('logs')
    if logs_dir.exists():
        count = 0
        for item in logs_dir.glob('**/*.log'):
            item.unlink()
            count += 1
        for item in logs_dir.glob('**/*.txt'):
            item.unlink()
            count += 1
        
        # Remove empty directories
        for dir_path in sorted(logs_dir.glob('**/'), reverse=True):
            if dir_path.is_dir() and not list(dir_path.iterdir()):
                dir_path.rmdir()
        
        print(f"✓ Deleted {count} log files")
    
    # Clean data cache
    cache_dir = Path('data_cache')
    if cache_dir.exists() and list(cache_dir.iterdir()):
        shutil.rmtree(cache_dir)
        cache_dir.mkdir()
        print("✓ Cleared data cache")

def cleanup_database():
    """Clean up all database training data"""
    db_url = load_database_url()
    
    if not db_url:
        print("❌ No DATABASE_URL found")
        return
    
    print("\n🗄️ Cleaning up database...")
    
    try:
        conn = psycopg2.connect(db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Get counts before cleanup
        tables = {
            'trade': 0,
            'training_metrics': 0,
            'trading_session': 0,
            'algorithm_config': 0,
            'market_data': 0
        }
        
        for table in tables:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                tables[table] = cur.fetchone()[0]
            except:
                pass
        
        # Clean up trading data (preserve market data)
        cur.execute("DELETE FROM trade")
        trades_deleted = cur.rowcount
        
        cur.execute("DELETE FROM training_metrics")
        metrics_deleted = cur.rowcount
        
        cur.execute("DELETE FROM trading_session")
        sessions_deleted = cur.rowcount
        
        cur.execute("DELETE FROM algorithm_config")
        configs_deleted = cur.rowcount
        
        print(f"✓ Deleted {sessions_deleted} training sessions")
        print(f"✓ Deleted {trades_deleted} trades")
        print(f"✓ Deleted {metrics_deleted} training metrics")
        print(f"✓ Deleted {configs_deleted} algorithm configs")
        print(f"✓ Preserved {tables['market_data']} market data records")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {e}")

def main(auto_confirm=False):
    print("🚀 AI Trading System - Complete Training Data Cleanup")
    print("=" * 60)
    print("This will remove ALL training data, models, and sessions!")
    print("Market data will be preserved.\n")
    
    if not auto_confirm:
        response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("Cleanup cancelled.")
            return
    
    cleanup_files()
    cleanup_database()
    
    print("\n✅ Cleanup complete!")
    print("The system is now ready for fresh training sessions.")

if __name__ == '__main__':
    import sys
    auto = '--auto' in sys.argv
    main(auto_confirm=auto)