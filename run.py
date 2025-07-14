#!/usr/bin/env python3
"""
Simple run script for the AI Trading System API

Usage:
    python run.py
"""

import os
import sys
import logging
from app import app, socketio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the Flask application with SocketIO support"""
    
    # Get port from environment or default
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("AI Trading System API")
    print("="*60)
    print(f"Starting server on http://0.0.0.0:{port}")
    print("\nAPI Endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /api/sessions         - List all sessions")
    print("  POST /api/start_training   - Start training")
    print("  POST /api/stop_training    - Stop training")
    print("  GET  /api/trades           - Get trades")
    print("  GET  /api/market_data      - Get market data")
    print("\nWebSocket Events:")
    print("  - performance_update")
    print("  - trade_update")
    print("  - session_update")
    print("="*60 + "\n")
    
    # Run the app with SocketIO
    try:
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        sys.exit(0)

if __name__ == '__main__':
    main()