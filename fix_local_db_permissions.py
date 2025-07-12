#!/usr/bin/env python3
"""
Fix database permission issues for local development
Run this script when you get "readonly database" errors
"""

import os
import sys
import stat
import sqlite3
from pathlib import Path

def fix_permissions():
    """Fix database file permissions"""
    print("🔧 Fixing database permissions...")
    
    # Find database path
    db_paths = [
        Path.home() / "PycharmProjects/ReinforcementTrader/instance/trading_system.db",
        Path.home() / "PycharmProjects/ReinforcementTrader/trading_system.db",
        Path("instance/trading_system.db"),
        Path("trading_system.db")
    ]
    
    db_path = None
    for path in db_paths:
        if path.exists():
            db_path = path
            break
    
    if not db_path:
        print("❌ Database file not found!")
        return False
        
    print(f"📁 Found database at: {db_path}")
    
    # Fix permissions on main database file
    try:
        os.chmod(db_path, 0o666)
        print(f"✓ Fixed permissions on {db_path}")
    except Exception as e:
        print(f"❌ Error fixing permissions on {db_path}: {e}")
        return False
    
    # Fix permissions on WAL and SHM files if they exist
    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")
    
    for path in [wal_path, shm_path]:
        if path.exists():
            try:
                os.chmod(path, 0o666)
                print(f"✓ Fixed permissions on {path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not fix permissions on {path}: {e}")
    
    # Test write access
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE IF NOT EXISTS test_write (id INTEGER)")
        conn.execute("DROP TABLE IF EXISTS test_write")
        conn.commit()
        conn.close()
        print("✓ Database is writable")
        return True
    except Exception as e:
        print(f"❌ Database still not writable: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Database Permission Fixer for Trading System")
    print("=" * 50)
    
    if fix_permissions():
        print("\n✅ Database permissions fixed successfully!")
        print("You can now run the trading system without readonly errors.")
    else:
        print("\n❌ Failed to fix database permissions.")
        print("Try running this script with sudo:")
        print("  sudo python fix_local_db_permissions.py")

if __name__ == "__main__":
    main()