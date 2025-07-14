#!/usr/bin/env python3
"""
Check comprehensive training status including data size, GPU usage, and session info
"""
import os
import torch
import subprocess
import psutil
from app import app
from extensions import db
from models import TradingSession, MarketData
from sqlalchemy import text
import json

def check_training_status():
    """Check comprehensive training status"""
    print("🔍 Comprehensive Training Status")
    print("=" * 80)
    
    # Check GPU usage
    print("\n🎮 GPU Status:")
    print("-" * 40)
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Total GPUs available: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3    # GB
            memory_total = props.total_memory / 1024**3                  # GB
            
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {memory_allocated:.2f}/{memory_reserved:.2f}/{memory_total:.2f} GB (allocated/reserved/total)")
            print(f"  Multi-processor count: {props.multi_processor_count}")
        
        # Check nvidia-smi for real-time usage
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                                   '--format=csv,noheader'], capture_output=True, text=True)
            if result.returncode == 0:
                print("\nReal-time GPU Usage (nvidia-smi):")
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        idx, name, util, mem_used, mem_total, temp = parts[:6]
                        print(f"  GPU {idx}: {util} utilization, Memory: {mem_used}/{mem_total}, Temp: {temp}")
        except:
            pass
    else:
        print("❌ No CUDA GPUs available")
    
    with app.app_context():
        # Check data size
        print("\n📊 Data Statistics:")
        print("-" * 40)
        
        # Total market data records
        total_records = db.session.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
        print(f"Total market data records: {total_records:,}")
        
        # Data by symbol
        symbol_data = db.session.execute(text("""
            SELECT symbol, COUNT(*) as count, 
                   MIN(timestamp) as earliest, 
                   MAX(timestamp) as latest
            FROM market_data
            GROUP BY symbol
        """))
        
        for row in symbol_data:
            print(f"\nSymbol: {row.symbol}")
            print(f"  Records: {row.count:,}")
            print(f"  Date range: {row.earliest} to {row.latest}")
        
        # Check active session details
        print("\n🚀 Active Training Session:")
        print("-" * 40)
        
        active_session = TradingSession.query.filter_by(status='active').first()
        if active_session:
            print(f"Session: {active_session.session_name} (ID: {active_session.id})")
            print(f"Algorithm: {active_session.algorithm_type}")
            print(f"Status: {active_session.status}")
            print(f"Progress: {active_session.current_episode}/{active_session.total_episodes} episodes")
            
            # Get algorithm config
            if active_session.algorithm_config:
                config = json.loads(active_session.algorithm_config) if isinstance(active_session.algorithm_config, str) else active_session.algorithm_config
                print(f"\nAlgorithm Configuration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
            
            # Check how much data is being used per episode
            print(f"\n📈 Data Usage per Episode:")
            print(f"  Window size: 60 time steps")
            print(f"  States per episode: 10")
            print(f"  Total data points per episode: 600 (60 × 10)")
            print(f"  Data coverage: {(600 / total_records * 100):.4f}% per episode")
            
            # Estimate time remaining
            if active_session.current_episode > 0:
                elapsed_time = (active_session.updated_at - active_session.start_time).total_seconds()
                time_per_episode = elapsed_time / active_session.current_episode
                remaining_episodes = active_session.total_episodes - active_session.current_episode
                estimated_time = remaining_episodes * time_per_episode
                
                print(f"\n⏱️  Time Estimation:")
                print(f"  Average time per episode: {time_per_episode:.2f} seconds")
                print(f"  Estimated time remaining: {estimated_time/3600:.2f} hours")
        else:
            print("❌ No active training session")
        
        # System resources
        print("\n💻 System Resources:")
        print("-" * 40)
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        print(f"CPU Usage: {cpu_percent}%")
        print(f"Memory Usage: {memory.percent}% ({memory.used/1024**3:.2f}/{memory.total/1024**3:.2f} GB)")
        
        # Check if training process is running
        print("\n🐍 Training Processes:")
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'] and proc.info['cmdline']:
                    cmdline = ' '.join(proc.info['cmdline'])
                    if 'run_local.py' in cmdline or 'START_HERE.py' in cmdline:
                        print(f"  PID {proc.info['pid']}: {cmdline[:80]}...")
            except:
                pass

def increase_episodes():
    """Show how to increase episodes"""
    print("\n\n📝 How to Increase Episodes:")
    print("=" * 80)
    print("\n1. Stop current training (Ctrl+C in run_local.py terminal)")
    print("\n2. Start new session with more episodes:")
    print('curl -X POST http://127.0.0.1:5000/api/start_training \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{')
    print('    "session_name": "Extended Training",')
    print('    "algorithm_type": "ANE_PPO",')
    print('    "total_episodes": 1000,')  # Increased episodes
    print('    "learning_rate": 0.0005,')
    print('    "clip_range": 0.3,')
    print('    "entropy_coef": 0.05,')
    print('    "n_steps": 512,')
    print('    "batch_size": 32')
    print('  }\'')
    print("\n3. Or modify the existing session (if you want to continue):")
    print("   - This requires updating the database directly")
    print("   - Not recommended while training is active")

if __name__ == "__main__":
    check_training_status()
    increase_episodes()