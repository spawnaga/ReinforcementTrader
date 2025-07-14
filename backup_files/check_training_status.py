#!/usr/bin/env python3
"""
Check comprehensive training status including data size, GPU usage, and session info
"""
import os
import torch
from sqlalchemy import create_engine, text
from datetime import datetime

def check_training_status():
    """Check comprehensive training status"""
    print("=== Training System Status Check ===")
    print(f"Time: {datetime.now()}")
    print("=" * 50)
    
    # 1. Check GPU availability
    print("\n1. GPU Status:")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   ✓ CUDA available with {gpu_count} GPU(s)")
        for i in range(gpu_count):
            print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
    else:
        print("   ✗ No CUDA GPUs available")
    
    # 2. Check database
    print("\n2. Database Check:")
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("   ✗ DATABASE_URL not set")
        return
        
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Check total data rows
            result = conn.execute(text("SELECT COUNT(*) FROM market_data"))
            data_count = result.scalar()
            print(f"   - Market data rows: {data_count:,}")
            
            # Check sessions
            result = conn.execute(text("""
                SELECT id, session_name, status, current_episode, total_episodes, algorithm_type
                FROM trading_session 
                ORDER BY id DESC 
                LIMIT 10
            """))
            sessions = result.fetchall()
            
            print(f"\n   Recent sessions (showing last 10):")
            for s in sessions:
                status_icon = "🟢" if s[2] == 'active' else "⚪"
                print(f"   {status_icon} Session {s[0]}: {s[1]}")
                print(f"      Status: {s[2]}, Algorithm: {s[5]}")
                print(f"      Progress: {s[3]}/{s[4]} episodes")
            
            # Check active sessions specifically
            result = conn.execute(text("SELECT COUNT(*) FROM trading_session WHERE status='active'"))
            active_count = result.scalar()
            print(f"\n   Active sessions: {active_count}")
            
            # Check trades by session
            result = conn.execute(text("""
                SELECT session_id, COUNT(*) as trade_count 
                FROM trade 
                GROUP BY session_id 
                ORDER BY session_id DESC 
                LIMIT 10
            """))
            trade_counts = result.fetchall()
            
            print(f"\n   Trades by session:")
            for tc in trade_counts:
                print(f"   - Session {tc[0]}: {tc[1]} trades")
                
    except Exception as e:
        print(f"   ✗ Database error: {e}")
    
    # 3. Check which environment is being used
    print("\n3. Trading Environment Check:")
    
    # Check if futures_env_realistic exists
    if os.path.exists('futures_env_realistic.py'):
        print("   ✓ futures_env_realistic.py found (realistic constraints)")
        
        # Check if it's being imported
        try:
            with open('gym_futures/envs/__init__.py', 'r') as f:
                content = f.read()
                if 'RealisticFuturesEnv' in content:
                    print("   ✓ RealisticFuturesEnv is registered")
                else:
                    print("   ✗ RealisticFuturesEnv not registered in __init__.py")
        except:
            pass
    else:
        print("   ✗ futures_env_realistic.py not found")
    
    print("\n4. Configuration Check:")
    print(f"   - Data directory: {'data/' if os.path.exists('data/') else 'NOT FOUND'}")
    print(f"   - Models directory: {'models/' if os.path.exists('models/') else 'NOT FOUND'}")
    print(f"   - Logs directory: {'logs/' if os.path.exists('logs/') else 'NOT FOUND'}")
    
    print("\n" + "=" * 50)

def increase_episodes():
    """Show how to increase episodes"""
    print("\nTo increase training episodes:")
    print("1. Update the session in the database:")
    print("   UPDATE trading_session SET total_episodes = 10000 WHERE id = 8;")
    print("\n2. Or restart training with more episodes")

if __name__ == "__main__":
    check_training_status()
    increase_episodes()