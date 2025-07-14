#!/usr/bin/env python3
"""
Diagnose why training is stuck
"""
import os
import sys
from app import app
from extensions import db
from models import TradingSession
from sqlalchemy import text
import subprocess

def diagnose_training():
    """Diagnose training issues"""
    print("🔍 Diagnosing Training Issues...")
    print("=" * 80)
    
    with app.app_context():
        # Check active sessions
        active_sessions = TradingSession.query.filter_by(status='active').all()
        print(f"\n📊 Active Sessions: {len(active_sessions)}")
        
        for session in active_sessions:
            print(f"\nSession ID {session.id}: {session.session_name}")
            print(f"  Status: {session.status}")
            print(f"  Started: {session.start_time}")
            print(f"  Current Episode: {session.current_episode}")
            
        # Check market data
        result = db.session.execute(text("SELECT COUNT(*) FROM market_data"))
        count = result.scalar()
        print(f"\n📈 Market Data Records: {count:,}")
        
        # Check log files
        print("\n📄 Checking Log Files:")
        log_dir = "logs"
        if os.path.exists(log_dir):
            for log_file in os.listdir(log_dir):
                if log_file.endswith('.log'):
                    file_path = os.path.join(log_dir, log_file)
                    size = os.path.getsize(file_path)
                    print(f"  - {log_file}: {size} bytes")
                    
                    # Check last few lines for errors
                    try:
                        with open(file_path, 'r') as f:
                            lines = f.readlines()
                            last_lines = lines[-20:] if len(lines) > 20 else lines
                            
                            errors_found = False
                            for line in last_lines:
                                if 'ERROR' in line or 'CRITICAL' in line:
                                    if not errors_found:
                                        print(f"\n  ⚠️  Recent errors in {log_file}:")
                                        errors_found = True
                                    print(f"    {line.strip()}")
                    except Exception as e:
                        print(f"    Could not read: {e}")
        
        # Check Python processes
        print("\n🐍 Python Processes:")
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            python_processes = [line for line in lines if 'python' in line and 'ReinforcementTrader' in line]
            for proc in python_processes[:5]:  # Show first 5
                cols = proc.split()
                if len(cols) > 10:
                    pid = cols[1]
                    cpu = cols[2]
                    mem = cols[3]
                    cmd = ' '.join(cols[10:])[:80]
                    print(f"  PID {pid}: CPU={cpu}%, MEM={mem}%, CMD={cmd}...")
        except:
            print("  Could not get process list")
        
        # Check GPU usage
        print("\n🎮 GPU Status:")
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        idx, name, util, mem_used, mem_total = parts[:5]
                        print(f"  GPU {idx}: {name} - Utilization: {util}, Memory: {mem_used}/{mem_total}")
            else:
                print("  Could not get GPU status")
        except:
            print("  nvidia-smi not available")
        
        print("\n💡 Suggestions:")
        print("1. Check the terminal where START_HERE.py is running for error messages")
        print("2. Look for 'ERROR' or 'Failed' messages in the logs")
        print("3. The training might be stuck in data loading or state creation")
        print("4. Try stopping this session and starting a new one with fewer episodes")
        
        print("\n🛠️  Commands to try:")
        print("# Stop the stuck session:")
        print("curl -X POST http://127.0.0.1:5000/api/sessions/1/stop")
        print("\n# Start a new test session with just 10 episodes:")
        print("curl -X POST http://127.0.0.1:5000/api/start_training \\")
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"session_name": "Quick Test", "algorithm_type": "ANE_PPO", "total_episodes": 10}\'')

if __name__ == "__main__":
    diagnose_training()