#!/usr/bin/env python3
"""
Local development startup script with proper worker configuration
This avoids conflicts between ib_insync and eventlet
"""

import subprocess
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main():
    print("\n🚀 Starting Revolutionary AI Trading System (Local Development)...")
    print("=" * 60)
    
    # Check for DATABASE_URL
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set!")
        print("\nTo set up PostgreSQL:")
        print("1. Install PostgreSQL")
        print("2. Create database and user:")
        print("   sudo -u postgres psql")
        print("   CREATE DATABASE reinforcement_trader;")
        print("   CREATE USER trader_user WITH PASSWORD 'your_password';")
        print("   GRANT ALL PRIVILEGES ON DATABASE reinforcement_trader TO trader_user;")
        print("3. Set environment variable:")
        print("   export DATABASE_URL='postgresql://trader_user:your_password@localhost:5432/reinforcement_trader'")
        print("\nFor detailed instructions, see POSTGRESQL_SETUP.md")
        print("=" * 60)
        sys.exit(1)
    
    print(f"✓ Using PostgreSQL database")
    
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