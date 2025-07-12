"""
Database utility functions for PostgreSQL operations and retry logic
"""
import time
import logging
from functools import wraps
from sqlalchemy.exc import OperationalError
from sqlalchemy import text
from flask import current_app

logger = logging.getLogger(__name__)



def retry_on_db_error(max_retries=3, delay=0.5, backoff=2):
    """
    Decorator to retry database operations on OperationalError for PostgreSQL
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
                    logger.warning(f"Database operation failed: {error_msg}, attempt {retries + 1}/{max_retries}")
                    
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
    Get information about the current PostgreSQL database connection
    """
    try:
        from app import db
        
        # Get database URI
        db_uri = current_app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')
        
        return {
            'type': 'PostgreSQL',
            'uri': db_uri,
            'pool_size': current_app.config.get('SQLALCHEMY_ENGINE_OPTIONS', {}).get('pool_size', 'Unknown'),
            'max_overflow': current_app.config.get('SQLALCHEMY_ENGINE_OPTIONS', {}).get('max_overflow', 'Unknown')
        }
            
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {'error': str(e)}

def ensure_db_writable():
    """
    Ensure the PostgreSQL database is accessible
    """
    try:
        from app import db
        
        # Test database connection
        db.session.execute(text("SELECT 1"))
        db.session.commit()
        
        return True
        
    except OperationalError as e:
        logger.error(f"Database operation failed: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        return False