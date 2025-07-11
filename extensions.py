"""
Flask extensions initialization
This file prevents circular imports by centralizing extension creation
"""
from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

# Initialize extensions
db = SQLAlchemy(model_class=Base)
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')