#!/usr/bin/env python3
"""
EASY STARTUP SCRIPT FOR THE TRADING SYSTEM

This script ensures the application runs correctly with WebSocket support.
Just run: python START_HERE.py
"""

import subprocess
import sys
import os

def main():
    print("\n🚀 Starting Revolutionary AI Trading System...")
    print("=" * 60)
    
    # Check if gunicorn is installed
    try:
        import gunicorn
        print("✓ Gunicorn is installed")
    except ImportError:
        print("× Gunicorn not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn"])
        print("✓ Gunicorn installed successfully")
    
    # Set environment variables if not set
    if not os.environ.get('DATABASE_URL'):
        # Create instance directory if it doesn't exist
        instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
        os.makedirs(instance_dir, exist_ok=True)
        
        # Use absolute path for database
        db_path = os.path.join(instance_dir, 'trading_system.db')
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        print(f"✓ Using SQLite database at: {db_path}")
    
    # Start the application
    print("\n🌐 Starting server on http://127.0.0.1:5000")
    print("=" * 60)
    print("\n⚡ The application is starting...")
    print("📊 Open your browser to http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run gunicorn with config file
        subprocess.run([
            "gunicorn",
            "--config", "gunicorn_config.py",
            "--bind", "127.0.0.1:5000",
            "main:app"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully")

if __name__ == "__main__":
    main()