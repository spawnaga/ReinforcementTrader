#!/usr/bin/env python3
"""
Simple local run script for the AI Trading System API without SocketIO
This version works without websocket support for easy local testing

Usage:
    python run_local.py
"""

import os
import sys
import logging
from app import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run the Flask application without SocketIO"""
    
    # Get port from environment or default
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("AI Trading System API (Local Version)")
    print("="*60)
    print(f"Starting server on http://localhost:{port}")
    print("\nAPI Endpoints:")
    print("  GET  /health              - Health check")
    print("  GET  /api/sessions         - List all sessions")
    print("  POST /api/start_training   - Start training")
    print("  POST /api/stop_training    - Stop training")
    print("  GET  /api/trades           - Get trades")
    print("  GET  /api/market_data      - Get market data")
    print("\nNote: WebSocket features are disabled in this version")
    print("="*60 + "\n")
    
    # Run the app without SocketIO
    try:
        app.run(
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