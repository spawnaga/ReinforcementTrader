#!/usr/bin/env python
"""
Local development runner that uses Gunicorn to avoid WebSocket issues
"""
import os
import sys
import subprocess

def main():
    """Run the application with Gunicorn for proper WebSocket support"""
    
    # Set environment variables for development
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Command to run Gunicorn
    cmd = [
        sys.executable, '-m', 'gunicorn',
        '--bind', '127.0.0.1:5000',
        '--worker-class', 'sync',
        '--workers', '1',
        '--threads', '4',
        '--timeout', '120',
        '--reload',
        '--log-level', 'debug',
        'main:app'
    ]
    
    print("Starting development server with Gunicorn...")
    print("This avoids WebSocket errors with Flask's development server")
    print("Access the app at: http://127.0.0.1:5000")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == '__main__':
    main()