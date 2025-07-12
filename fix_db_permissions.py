#!/usr/bin/env python3
"""
Utility script to diagnose and fix database permission issues
Run this script when experiencing "readonly database" errors
"""
import os
import sys
import stat
import sqlite3
from pathlib import Path

def check_database_permissions(db_path):
    """Check and report database file permissions"""
    print(f"\n🔍 Checking database: {db_path}")
    
    # Check if file exists
    if not os.path.exists(db_path):
        print(f"❌ Database file does not exist: {db_path}")
        return False
    
    # Check file permissions
    file_stat = os.stat(db_path)
    mode = file_stat.st_mode
    
    print(f"📊 File permissions: {oct(stat.S_IMODE(mode))}")
    print(f"👤 Owner: {file_stat.st_uid}")
    print(f"👥 Group: {file_stat.st_gid}")
    
    # Check if writable
    if os.access(db_path, os.W_OK):
        print("✅ Database is writable by current user")
    else:
        print("❌ Database is NOT writable by current user")
        return False
    
    # Check directory permissions
    db_dir = os.path.dirname(db_path)
    if os.access(db_dir, os.W_OK):
        print("✅ Database directory is writable")
    else:
        print("❌ Database directory is NOT writable")
        return False
    
    # Check WAL and SHM files
    for suffix in ['-wal', '-shm']:
        wal_path = db_path + suffix
        if os.path.exists(wal_path):
            if os.access(wal_path, os.W_OK):
                print(f"✅ {suffix} file is writable")
            else:
                print(f"❌ {suffix} file is NOT writable")
                return False
    
    return True

def fix_database_permissions(db_path):
    """Attempt to fix database permissions"""
    print(f"\n🔧 Attempting to fix permissions for: {db_path}")
    
    try:
        # Fix main database file
        if os.path.exists(db_path):
            current_mode = os.stat(db_path).st_mode
            new_mode = current_mode | stat.S_IWUSR | stat.S_IRUSR
            os.chmod(db_path, new_mode)
            print(f"✅ Fixed permissions for main database file")
        
        # Fix directory
        db_dir = os.path.dirname(db_path)
        if os.path.exists(db_dir):
            dir_mode = os.stat(db_dir).st_mode
            new_dir_mode = dir_mode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR
            os.chmod(db_dir, new_dir_mode)
            print(f"✅ Fixed permissions for database directory")
        
        # Fix WAL and SHM files
        for suffix in ['-wal', '-shm']:
            wal_path = db_path + suffix
            if os.path.exists(wal_path):
                wal_mode = os.stat(wal_path).st_mode
                new_wal_mode = wal_mode | stat.S_IWUSR | stat.S_IRUSR
                os.chmod(wal_path, new_wal_mode)
                print(f"✅ Fixed permissions for {suffix} file")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing permissions: {e}")
        print("\n💡 Try running with sudo: sudo python fix_db_permissions.py")
        return False

def test_database_write(db_path):
    """Test if we can write to the database"""
    print(f"\n🧪 Testing database write access...")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Try to create a test table and write to it
        cursor.execute("CREATE TABLE IF NOT EXISTS permission_test (id INTEGER PRIMARY KEY, test TEXT)")
        cursor.execute("INSERT INTO permission_test (test) VALUES ('write test')")
        conn.commit()
        
        # Clean up
        cursor.execute("DROP TABLE permission_test")
        conn.commit()
        conn.close()
        
        print("✅ Database write test successful!")
        return True
        
    except sqlite3.OperationalError as e:
        print(f"❌ Database write test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Trading System Database Permission Fixer")
    print("=" * 50)
    
    # Determine database path
    db_paths = [
        "instance/trading_system.db",
        "/home/alex/PycharmProjects/ReinforcementTrader/instance/trading_system.db",
        os.path.join(os.getcwd(), "instance", "trading_system.db")
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        # Try to find any .db file
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".db"):
                    db_path = os.path.join(root, file)
                    break
            if db_path:
                break
    
    if not db_path:
        print("❌ No database file found!")
        print("\n💡 Please specify the database path as an argument:")
        print("   python fix_db_permissions.py /path/to/database.db")
        sys.exit(1)
    
    # Check command line argument
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    
    print(f"\n📁 Database path: {db_path}")
    
    # Check permissions
    if not check_database_permissions(db_path):
        # Try to fix
        if fix_database_permissions(db_path):
            # Re-check
            print("\n🔍 Re-checking permissions after fix...")
            check_database_permissions(db_path)
            
            # Test write
            test_database_write(db_path)
        else:
            print("\n❌ Failed to fix permissions automatically")
            print("\n💡 Manual fix suggestions:")
            print(f"   1. chmod 664 {db_path}")
            print(f"   2. chmod 775 {os.path.dirname(db_path)}")
            print(f"   3. chown $USER:$USER {db_path}")
            print(f"   4. If using WSL, check Windows file permissions")
    else:
        # Test write anyway
        test_database_write(db_path)
    
    print("\n✨ Done!")

if __name__ == "__main__":
    main()