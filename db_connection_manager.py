"""
Database Connection Manager for SQLite Concurrent Access

This module provides a thread-safe connection manager that ensures
only one writer accesses the SQLite database at a time.
"""

import threading
import time
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event
from sqlalchemy.pool import StaticPool

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """Thread-safe database connection manager for SQLite"""
    
    def __init__(self, database_uri):
        self.database_uri = database_uri
        self.write_lock = threading.Lock()
        self._local = threading.local()
        
        # Create engine with special SQLite settings
        if 'sqlite' in database_uri:
            self.engine = create_engine(
                database_uri,
                connect_args={
                    'check_same_thread': False,
                    'timeout': 30.0,  # 30 second timeout
                    'isolation_level': None  # Autocommit mode
                },
                poolclass=StaticPool,  # Use single connection pool
                echo=False
            )
            
            # Configure SQLite for better concurrent access
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.execute("PRAGMA mmap_size=30000000000")
                cursor.close()
        else:
            # Non-SQLite database
            self.engine = create_engine(
                database_uri,
                pool_size=20,
                max_overflow=0,
                pool_pre_ping=True,
                pool_recycle=300
            )
    
    @contextmanager
    def get_connection(self, write_operation=False):
        """Get a database connection with proper locking for write operations"""
        if write_operation and 'sqlite' in self.database_uri:
            # Acquire write lock for SQLite
            with self.write_lock:
                conn = self.engine.connect()
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
        else:
            # Read operation or non-SQLite database
            conn = self.engine.connect()
            try:
                yield conn
            finally:
                conn.close()
    
    def execute_with_retry(self, operation, max_retries=3, delay=1.0, write_operation=False):
        """Execute a database operation with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                with self.get_connection(write_operation=write_operation) as conn:
                    return operation(conn)
            except Exception as e:
                last_error = e
                if 'database is locked' in str(e) or 'attempt to write a readonly database' in str(e):
                    logger.warning(f"Database locked on attempt {attempt + 1}, retrying in {delay} seconds...")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise
        
        # If we get here, all retries failed
        logger.error(f"All {max_retries} attempts failed")
        raise last_error

# Global instance
_db_manager = None

def get_db_manager():
    """Get the global database connection manager"""
    global _db_manager
    if _db_manager is None:
        from flask import current_app
        _db_manager = DatabaseConnectionManager(current_app.config['SQLALCHEMY_DATABASE_URI'])
    return _db_manager

@contextmanager
def get_db_connection(write_operation=False):
    """Convenience function to get a database connection"""
    manager = get_db_manager()
    with manager.get_connection(write_operation=write_operation) as conn:
        yield conn