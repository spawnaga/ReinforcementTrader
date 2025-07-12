"""
Database utility functions for handling SQLite permissions and retry logic
"""
import os
import stat
import time
import logging
from functools import wraps
from sqlalchemy.exc import OperationalError
from flask import current_app

logger = logging.getLogger(__name__)

def check_and_fix_db_permissions(db_path):
    """
    Check and fix database file permissions to ensure it's writable
    """
    try:
        # Check if database file exists
        if not os.path.exists(db_path):
            logger.warning(f"Database file does not exist: {db_path}")
            return False
            
        # Get current permissions
        current_perms = os.stat(db_path).st_mode
        
        # Check if file is writable
        if not os.access(db_path, os.W_OK):
            logger.warning(f"Database file is not writable: {db_path}")
            
            # Try to make it writable
            try:
                os.chmod(db_path, current_perms | stat.S_IWUSR | stat.S_IWGRP)
                logger.info(f"Fixed database permissions for: {db_path}")
            except Exception as e:
                logger.error(f"Could not fix database permissions: {e}")
                return False
                
        # Also check the directory permissions
        db_dir = os.path.dirname(db_path)
        if not os.access(db_dir, os.W_OK):
            logger.warning(f"Database directory is not writable: {db_dir}")
            try:
                dir_perms = os.stat(db_dir).st_mode
                os.chmod(db_dir, dir_perms | stat.S_IWUSR | stat.S_IWGRP)
                logger.info(f"Fixed directory permissions for: {db_dir}")
            except Exception as e:
                logger.error(f"Could not fix directory permissions: {e}")
                return False
                
        # Check WAL and SHM files if they exist
        for suffix in ['-wal', '-shm']:
            wal_path = db_path + suffix
            if os.path.exists(wal_path) and not os.access(wal_path, os.W_OK):
                try:
                    wal_perms = os.stat(wal_path).st_mode
                    os.chmod(wal_path, wal_perms | stat.S_IWUSR | stat.S_IWGRP)
                    logger.info(f"Fixed permissions for: {wal_path}")
                except Exception as e:
                    logger.error(f"Could not fix WAL/SHM permissions: {e}")
                    
        return True
        
    except Exception as e:
        logger.error(f"Error checking database permissions: {e}")
        return False

def retry_on_db_error(max_retries=3, delay=0.5, backoff=2):
    """
    Decorator to retry database operations on OperationalError
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    error_msg = str(e)
                    
                    if "readonly database" in error_msg:
                        logger.warning(f"Database is readonly, attempt {retries + 1}/{max_retries}")
                        
                        # Try to fix permissions
                        if hasattr(current_app, 'config'):
                            db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
                            if 'sqlite:///' in db_uri:
                                db_path = db_uri.replace('sqlite:///', '')
                                check_and_fix_db_permissions(db_path)
                    
                    elif "database is locked" in error_msg:
                        logger.warning(f"Database is locked, attempt {retries + 1}/{max_retries}")
                    
                    else:
                        # For other operational errors, re-raise immediately
                        raise
                    
                    retries += 1
                    if retries < max_retries:
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"Max retries reached for {func.__name__}")
                        raise
                        
                except Exception as e:
                    # For non-operational errors, raise immediately
                    raise
                    
        return wrapper
    return decorator

def get_db_info():
    """
    Get information about the current database connection
    """
    try:
        from app import db
        
        # Get database URI
        db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')
        
        # Check if it's SQLite
        if 'sqlite:///' in db_uri:
            db_path = db_uri.replace('sqlite:///', '')
            
            info = {
                'type': 'SQLite',
                'path': db_path,
                'exists': os.path.exists(db_path),
                'writable': os.access(db_path, os.W_OK) if os.path.exists(db_path) else False,
                'size': os.path.getsize(db_path) if os.path.exists(db_path) else 0
            }
            
            # Check WAL mode
            try:
                result = db.session.execute("PRAGMA journal_mode").fetchone()
                info['journal_mode'] = result[0] if result else 'unknown'
            except:
                info['journal_mode'] = 'error'
                
            return info
        else:
            return {
                'type': 'PostgreSQL/MySQL',
                'uri': db_uri
            }
            
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {'error': str(e)}

def ensure_db_writable():
    """
    Ensure the database is writable before operations
    """
    try:
        from app import db
        
        # Test write operation
        db.session.execute("SELECT 1")
        db.session.commit()
        
        return True
        
    except OperationalError as e:
        error_msg = str(e)
        
        if "readonly database" in error_msg:
            logger.error("Database is readonly - checking permissions")
            
            db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', '')
            if 'sqlite:///' in db_uri:
                db_path = db_uri.replace('sqlite:///', '')
                if check_and_fix_db_permissions(db_path):
                    # Try again after fixing permissions
                    try:
                        db.session.execute("SELECT 1")
                        db.session.commit()
                        return True
                    except:
                        pass
                        
        return False
        
    except Exception as e:
        logger.error(f"Error ensuring database is writable: {e}")
        return False