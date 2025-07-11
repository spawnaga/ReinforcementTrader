#!/usr/bin/env python3
"""
Simple runner that works in any environment (PyCharm, Terminal, etc.)
Just run: python run_app.py
"""

import os
import sys
import subprocess

# Ensure we're in the right directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Create instance directory
os.makedirs('instance', exist_ok=True)

# Set database path
db_path = os.path.join(os.getcwd(), 'instance', 'trading_system.db')
os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'

print("\n🚀 Starting Trading System...")
print(f"📁 Database: {db_path}")
print("🌐 Server: http://127.0.0.1:5000\n")

# Try to run with gunicorn
try:
    subprocess.run([sys.executable, "-m", "gunicorn", "--bind", "127.0.0.1:5000", "--reload", "main:app"])
except:
    print("Installing gunicorn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gunicorn"])
    subprocess.run([sys.executable, "-m", "gunicorn", "--bind", "127.0.0.1:5000", "--reload", "main:app"])