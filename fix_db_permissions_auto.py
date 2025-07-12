#!/usr/bin/env python3
"""
Automatic Database Permission Fixer

This script automatically fixes database permissions whenever they get reset.
Run this in the background to prevent "readonly database" errors.
"""

import os
import sys
import time
import stat
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_database_files():
    """Find all SQLite database files in the project"""
    db_files = []
    
    # Check instance directory
    instance_dir = Path('instance')
    if instance_dir.exists():
        for db_file in instance_dir.glob('*.db'):
            db_files.append(db_file)
    
    # Check root directory
    for db_file in Path('.').glob('*.db'):
        db_files.append(db_file)
    
    return db_files

def fix_permissions(file_path):
    """Fix permissions for a single file"""
    try:
        # Set read/write permissions for everyone (666)
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | 
                          stat.S_IRGRP | stat.S_IWGRP | 
                          stat.S_IROTH | stat.S_IWOTH)
        return True
    except Exception as e:
        logger.error(f"Failed to fix permissions for {file_path}: {e}")
        return False

def fix_database_permissions():
    """Fix permissions for all database files and their associated files"""
    fixed_count = 0
    
    db_files = find_database_files()
    
    if not db_files:
        logger.warning("No database files found")
        return 0
    
    for db_file in db_files:
        # Fix main database file
        if db_file.exists():
            if fix_permissions(db_file):
                logger.info(f"Fixed permissions for: {db_file}")
                fixed_count += 1
            
            # Fix WAL file
            wal_file = Path(f"{db_file}-wal")
            if wal_file.exists():
                if fix_permissions(wal_file):
                    logger.info(f"Fixed permissions for: {wal_file}")
                    fixed_count += 1
            
            # Fix SHM file
            shm_file = Path(f"{db_file}-shm")
            if shm_file.exists():
                if fix_permissions(shm_file):
                    logger.info(f"Fixed permissions for: {shm_file}")
                    fixed_count += 1
            
            # Fix journal file (if exists)
            journal_file = Path(f"{db_file}-journal")
            if journal_file.exists():
                if fix_permissions(journal_file):
                    logger.info(f"Fixed permissions for: {journal_file}")
                    fixed_count += 1
    
    return fixed_count

def monitor_and_fix(interval=30):
    """Monitor and fix database permissions continuously"""
    logger.info(f"Starting database permission monitor (checking every {interval} seconds)")
    
    while True:
        try:
            fixed = fix_database_permissions()
            if fixed > 0:
                logger.info(f"Fixed {fixed} file(s)")
            else:
                logger.debug("All database files have correct permissions")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("Stopping database permission monitor")
            break
        except Exception as e:
            logger.error(f"Error in monitor loop: {e}")
            time.sleep(interval)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--once':
            # Run once and exit
            fixed = fix_database_permissions()
            logger.info(f"Fixed {fixed} file(s)")
            sys.exit(0)
        elif sys.argv[1] == '--help':
            print("Usage: python fix_db_permissions_auto.py [--once|--monitor]")
            print("  --once    Fix permissions once and exit")
            print("  --monitor Run continuously and fix permissions (default)")
            sys.exit(0)
    
    # Default: run in monitor mode
    try:
        monitor_and_fix()
    except KeyboardInterrupt:
        logger.info("Exiting...")

if __name__ == '__main__':
    main()