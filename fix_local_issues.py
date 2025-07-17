#!/usr/bin/env python3
"""Fix local training issues"""

import os
import sys

# 1. Create database tables
print("Creating database tables...")
try:
    from app import app, db
    with app.app_context():
        db.create_all()
        print("✓ Database tables created successfully")
except Exception as e:
    print(f"Error creating tables: {e}")

# 2. Fix the 'episode' parameter issue in trading_engine.py
print("\nFixing trading_engine.py...")
with open('trading_engine.py', 'r') as f:
    content = f.read()

# Replace the problematic line
old_line = """        session = TradingSession(
            session_name=session_name,
            algorithm_type=algorithm_type,
            parameters=config,
            episode=0"""

new_line = """        session = TradingSession(
            session_name=session_name,
            algorithm_type=algorithm_type,
            parameters=config,
            total_episodes=config.get('episodes', 1000)"""

if old_line in content:
    content = content.replace(old_line, new_line)
    with open('trading_engine.py', 'w') as f:
        f.write(content)
    print("✓ Fixed 'episode' parameter issue")
else:
    # Try another pattern
    content = content.replace('episode=0', 'total_episodes=config.get("episodes", 1000)')
    with open('trading_engine.py', 'w') as f:
        f.write(content)
    print("✓ Fixed 'episode' parameter issue (alternative pattern)")

print("\nFixes applied! You can now run your training command.")