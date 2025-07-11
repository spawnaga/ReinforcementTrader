#!/usr/bin/env python
"""Test script to verify circular import fixes"""

print("Testing imports...")

try:
    print("1. Importing Flask app...")
    from app import app
    print("✓ App imported successfully")
    
    print("\n2. Importing extensions...")
    from extensions import db, socketio
    print("✓ Extensions imported successfully")
    
    print("\n3. Importing models...")
    from models import TradingSession, Trade, MarketData
    print("✓ Models imported successfully")
    
    print("\n4. Importing trading engine...")
    from trading_engine import TradingEngine
    print("✓ Trading engine imported successfully")
    
    print("\n5. Importing routes...")
    import routes
    print("✓ Routes imported successfully")
    
    print("\n6. Importing websocket handler...")
    import websocket_handler
    print("✓ WebSocket handler imported successfully")
    
    print("\n7. Importing data manager...")
    from data_manager import DataManager
    print("✓ Data manager imported successfully")
    
    print("\n✅ All imports successful! No circular import errors.")
    print("\nYou can now run the app with:")
    print("python app.py")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("There's still a circular import issue.")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")