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
        os.environ['DATABASE_URL'] = 'sqlite:///instance/trading_system.db'
        print("✓ Using SQLite database")
    
    # Start the application
    print("\n🌐 Starting server on http://127.0.0.1:5000")
    print("=" * 60)
    print("\n⚡ The application is starting...")
    print("📊 Open your browser to http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run gunicorn with proper settings
        subprocess.run([
            "gunicorn",
            "--bind", "127.0.0.1:5000",
            "--reload",
            "--worker-class", "sync",
            "--workers", "1",
            "--timeout", "120",
            "main:app"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully")

if __name__ == "__main__":
    main()