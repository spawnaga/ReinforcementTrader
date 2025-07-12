import os
import logging
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from extensions import db, socketio

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure the database
# Ensure instance directory exists
instance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
os.makedirs(instance_path, exist_ok=True)

# Set database URI with absolute path
default_db_path = os.path.join(instance_path, 'trading_system.db')
database_uri = os.environ.get("DATABASE_URL", f"sqlite:///{default_db_path}")
app.config["SQLALCHEMY_DATABASE_URI"] = database_uri

# Use proper SQLite configuration for concurrent access
if 'sqlite' in database_uri:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        'pool_pre_ping': True,
        'pool_size': 1,  # SQLite only allows one writer at a time
        'max_overflow': 0,  # Don't create extra connections
        'connect_args': {
            'check_same_thread': False,
            'timeout': 30,  # Increase timeout to 30 seconds
            'isolation_level': None  # Use autocommit mode
        }
    }
else:
    # PostgreSQL configuration optimized for multi-GPU training
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
        "pool_size": 30,        # Increased for 4 GPUs + threads
        "max_overflow": 20,     # Allow overflow connections
        "pool_timeout": 30,     # Connection timeout
        "echo_pool": True       # Log pool events for debugging
    }
    logging.info("PostgreSQL configured for high-concurrency multi-GPU training")

# Initialize extensions
db.init_app(app)
socketio.init_app(app)

# Initialize trading engine before importing routes to avoid circular imports
from trading_engine import TradingEngine

trading_engine = TradingEngine()

with app.app_context():
    # Create all tables
    db.create_all()

    # Enable WAL mode for SQLite to handle concurrent access better
    if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']:
        try:
            from sqlalchemy import text

            with db.engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.execute(text("PRAGMA busy_timeout=30000"))  # 30 seconds
                conn.execute(text("PRAGMA synchronous=NORMAL"))
                conn.commit()
                logging.info("SQLite WAL mode enabled for better concurrent access")
        except Exception as e:
            logging.warning(f"Could not enable WAL mode: {e}")

# Import routes after all setup (this registers them with the app)
from routes import *