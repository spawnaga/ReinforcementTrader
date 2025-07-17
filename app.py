import os
import logging
from dotenv import load_dotenv
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create database base class
class Base(DeclarativeBase):
    pass

# Initialize extensions
db = SQLAlchemy(model_class=Base)
# Initialize socketio after app creation to avoid import issues
socketio = None

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
# Initialize socketio after app creation
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

with app.app_context():
    # Create all tables
    db.create_all()

    # PostgreSQL doesn't need special configurations like SQLite WAL mode
    logging.info("PostgreSQL database ready for concurrent operations")

# Import routes after all setup (this registers them with the app)
from routes import *