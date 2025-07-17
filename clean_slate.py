#!/usr/bin/env python3
"""
Clean Slate - Remove all training progress and start fresh
This script will delete all training data, logs, models, and database records
"""
import os
import shutil
from pathlib import Path
import psycopg2
from datetime import datetime

def clean_slate():
    """Remove all training artifacts and start fresh"""
    print("="*60)
    print("CLEAN SLATE - Start Fresh")
    print("="*60)
    print("\nThis will DELETE:")
    print("  • All trained models in models/")
    print("  • All logs in logs/")
    print("  • All processed data in data/processed/")
    print("  • All cached data in data_cache/")
    print("  • All trading data in trading_data/")
    print("  • All database tables (if PostgreSQL available)")
    print("  • All temporary files and caches")
    print("\n⚠️  This action cannot be undone!")
    
    response = input("\nAre you sure you want to start completely fresh? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled. Nothing was deleted.")
        return
    
    print("\nStarting cleanup...")
    
    # 1. Delete all models
    models_dir = Path("models")
    if models_dir.exists():
        count = len(list(models_dir.glob("*.pt"))) + len(list(models_dir.glob("*.pth")))
        shutil.rmtree(models_dir)
        models_dir.mkdir(exist_ok=True)
        print(f"✓ Deleted {count} model files")
    else:
        models_dir.mkdir(exist_ok=True)
        print("✓ Created models/ directory")
    
    # 2. Delete all logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        # Count log files
        log_count = sum(1 for _ in logs_dir.rglob("*.log"))
        shutil.rmtree(logs_dir)
        logs_dir.mkdir(exist_ok=True)
        print(f"✓ Deleted {log_count} log files")
    else:
        logs_dir.mkdir(exist_ok=True)
        print("✓ Created logs/ directory")
    
    # 3. Delete processed data
    processed_data = Path("data/processed")
    if processed_data.exists():
        file_count = len(list(processed_data.glob("*")))
        shutil.rmtree(processed_data)
        processed_data.mkdir(parents=True, exist_ok=True)
        print(f"✓ Deleted {file_count} processed data files")
    else:
        processed_data.mkdir(parents=True, exist_ok=True)
        print("✓ Created data/processed/ directory")
    
    # 4. Delete data cache
    cache_dir = Path("data_cache")
    if cache_dir.exists():
        cache_count = len(list(cache_dir.glob("*")))
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        print(f"✓ Deleted {cache_count} cached files")
    else:
        cache_dir.mkdir(exist_ok=True)
        print("✓ Created data_cache/ directory")
    
    # 5. Delete trading data
    trading_dir = Path("trading_data")
    if trading_dir.exists():
        trading_count = len(list(trading_dir.glob("*")))
        shutil.rmtree(trading_dir)
        trading_dir.mkdir(exist_ok=True)
        print(f"✓ Deleted {trading_count} trading data files")
    else:
        trading_dir.mkdir(exist_ok=True)
        print("✓ Created trading_data/ directory")
    
    # 6. Clean Python cache
    pycache_count = 0
    for pycache in Path(".").rglob("__pycache__"):
        shutil.rmtree(pycache)
        pycache_count += 1
    print(f"✓ Deleted {pycache_count} Python cache directories")
    
    # 7. Delete database tables (if PostgreSQL available)
    try:
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
            conn = psycopg2.connect(database_url)
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()
            
            # Drop all training-related tables
            tables_to_drop = [
                'algorithm_metrics',
                'position_snapshots', 
                'trades',
                'episode_metrics',
                'learning_progress',
                'model_checkpoints',
                'training_sessions',
                'market_data',
                'trading_session',
                'trade',
                'training_metric',
                'algorithm_config'
            ]
            
            dropped_count = 0
            for table in tables_to_drop:
                try:
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    dropped_count += 1
                except:
                    pass
            
            # Drop views
            views_to_drop = ['session_summary', 'recent_performance']
            for view in views_to_drop:
                try:
                    cur.execute(f"DROP VIEW IF EXISTS {view} CASCADE")
                except:
                    pass
            
            conn.close()
            print(f"✓ Dropped {dropped_count} database tables")
    except Exception as e:
        print(f"ℹ PostgreSQL cleanup skipped (not configured)")
    
    # 8. Create .gitkeep files to preserve directory structure
    for dir_path in [models_dir, logs_dir, processed_data, cache_dir, trading_dir]:
        (dir_path / ".gitkeep").touch()
    
    print("\n" + "="*60)
    print("✓ CLEANUP COMPLETE - Fresh Start Ready!")
    print("="*60)
    print("\nNext steps:")
    print("1. Place your raw data file (e.g., NQ futures data) in data/")
    print("2. Run data preparation: python prepare_data.py")
    print("3. Start training: ./start_training.sh")
    print("\nYour project is now completely clean and ready for a fresh start!")

if __name__ == "__main__":
    clean_slate()