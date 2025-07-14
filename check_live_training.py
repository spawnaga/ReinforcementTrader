#!/usr/bin/env python3
"""
Check what's happening in live training
"""
import os
import time
from app import app
from extensions import db
from models import TradingSession, Trade
from sqlalchemy import text

def check_live_training():
    """Monitor live training activity"""
    with app.app_context():
        print("🔍 Live Training Monitor")
        print("=" * 60)
        
        # Get active session
        session = TradingSession.query.filter_by(status='active').first()
        if not session:
            print("❌ No active training session!")
            print("\nStart training with:")
            print("python run_local.py")
            print("\nThen in another terminal:")
            print('curl -X POST http://127.0.0.1:5000/api/start_training \\')
            print('  -H "Content-Type: application/json" \\')
            print('  -d \'{"session_name": "Test Run", "algorithm_type": "ANE_PPO", "total_episodes": 50}\'')
            return
        
        print(f"\n📊 Active Session: {session.session_name} (ID: {session.id})")
        print(f"   Started: {session.start_time}")
        print(f"   Episodes: {session.current_episode}")
        
        # Check recent trades
        recent_trades = db.session.execute(text("""
            SELECT id, entry_time, exit_time, entry_price, exit_price, profit_loss
            FROM trade
            WHERE session_id = :session_id
            ORDER BY id DESC
            LIMIT 10
        """), {"session_id": session.id})
        
        trades = list(recent_trades)
        print(f"\n📈 Recent Trades (Total: {len(trades)}):")
        
        open_trades = 0
        closed_trades = 0
        
        for trade in trades:
            if trade.exit_time:
                status = "CLOSED"
                closed_trades += 1
            else:
                status = "OPEN"
                open_trades += 1
            
            print(f"   Trade {trade.id}: {status}")
            print(f"     Entry: ${trade.entry_price:.2f} at {trade.entry_time}")
            if trade.exit_time:
                print(f"     Exit: ${trade.exit_price:.2f} at {trade.exit_time}")
                print(f"     P&L: ${trade.profit_loss:.2f}")
        
        print(f"\n📊 Trade Summary:")
        print(f"   Open trades: {open_trades}")
        print(f"   Closed trades: {closed_trades}")
        
        # Check log file for recent actions
        log_file = f"logs/trading_{session.session_name.replace(' ', '_')}_{session.id}.log"
        if os.path.exists(log_file):
            print(f"\n📄 Recent Actions from {log_file}:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                recent = lines[-20:] if len(lines) > 20 else lines
                
                actions = {"BUY": 0, "SELL": 0, "HOLD": 0}
                for line in recent:
                    for action in actions:
                        if f"ACTION: {action}" in line:
                            actions[action] += 1
                
                print(f"   Last 20 lines: BUY={actions['BUY']}, SELL={actions['SELL']}, HOLD={actions['HOLD']}")
                
                # Show last few action lines
                action_lines = [l for l in recent if "ACTION:" in l]
                if action_lines:
                    print("\n   Last 3 actions:")
                    for line in action_lines[-3:]:
                        print(f"     {line.strip()}")

if __name__ == "__main__":
    check_live_training()