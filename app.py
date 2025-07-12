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
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", f"sqlite:///{default_db_path}")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize extensions
db.init_app(app)
socketio.init_app(app)

# Initialize trading engine before importing routes to avoid circular imports
from trading_engine import TradingEngine
trading_engine = TradingEngine()

with app.app_context():
    # Import models and routes
    import models
    import routes
    import websocket_handler
    # import test_routes  # Commented out - test_training already exists
    
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
