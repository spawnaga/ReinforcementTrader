import os
import logging
from dotenv import load_dotenv
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from extensions import db, socketio

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure PostgreSQL database
database_uri = os.environ.get("DATABASE_URL")
if not database_uri:
    raise ValueError("DATABASE_URL environment variable must be set for PostgreSQL connection")

app.config["SQLALCHEMY_DATABASE_URI"] = database_uri

# PostgreSQL configuration optimized for multi-GPU training
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "pool_size": 30,        # Increased for 4 GPUs + threads
    "max_overflow": 20,     # Allow overflow connections
    "pool_timeout": 30,     # Connection timeout
    "echo_pool": False      # Disable pool logging in production
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

    # PostgreSQL doesn't need special configurations like SQLite WAL mode
    logging.info("PostgreSQL database ready for concurrent operations")

# Import routes after all setup (this registers them with the app)
from routes import *