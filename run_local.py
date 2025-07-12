#!/usr/bin/env python3
"""
Local development startup script with proper worker configuration
This avoids conflicts between ib_insync and eventlet
"""

import subprocess
import sys
import os

def main():
    print("\n🚀 Starting Revolutionary AI Trading System (Local Development)...")
    print("=" * 60)
    
    # Set environment variables if not set
    if not os.environ.get('DATABASE_URL'):
        # Create instance directory if it doesn't exist
        instance_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance')
        os.makedirs(instance_dir, exist_ok=True)
        
        # Use absolute path for database
        db_path = os.path.join(instance_dir, 'trading_system.db')
        os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
        print(f"✓ Using SQLite database at: {db_path}")
    
    # Start the application with threaded worker (better for ib_insync compatibility)
    print("\n🌐 Starting server on http://127.0.0.1:5000")
    print("✓ Using threaded worker for better compatibility")
    print("=" * 60)
    print("\n⚡ The application is starting...")
    print("📊 Open your browser to http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run gunicorn with gthread worker
        subprocess.run([
            "gunicorn",
            "--bind", "127.0.0.1:5000",
            "--worker-class", "gthread",  # Better than sync for WebSockets
            "--workers", "1",
            "--threads", "4",  # Multiple threads for concurrent requests
            "--timeout", "300",
            "--keep-alive", "5",
            "--reload",
            "main:app"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully")

if __name__ == "__main__":
    main()