#!/usr/bin/env python3
"""
Remote-accessible startup script for the trading system
Allows connections from other machines on the network
"""

import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    print("\n🚀 Starting Revolutionary AI Trading System (Remote Access Mode)...")
    print("=" * 60)
    
    # Check for DATABASE_URL
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set!")
        print("\nPlease set up PostgreSQL first. See POSTGRESQL_SETUP.md")
        sys.exit(1)
    
    print(f"✓ Using PostgreSQL database")
    
    # Get the host IP to bind to
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = os.environ.get('FLASK_PORT', '5000')
    
    # Start the application with network-accessible binding
    print(f"\n🌐 Starting server on http://{host}:{port}")
    print("✓ Server will be accessible from other machines on the network")
    print("=" * 60)
    print("\n⚡ The application is starting...")
    print(f"📊 Local access: http://localhost:{port}")
    print(f"📊 Network access: http://YOUR_IP:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        # Run gunicorn with network binding
        subprocess.run([
            "gunicorn",
            "--bind", f"{host}:{port}",
            "--worker-class", "gthread",
            "--workers", "1",
            "--threads", "4",
            "--timeout", "300",
            "--keep-alive", "5",
            "--reload",
            "main:app"
        ])
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped gracefully")

if __name__ == "__main__":
    # Set debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    main()