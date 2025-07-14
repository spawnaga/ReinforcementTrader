#!/usr/bin/env python3
"""
Diagnose why the agent isn't making trades
"""
import os
from app import app
from extensions import db
from models import TradingSession, Trade
from sqlalchemy import text
import json

def diagnose_no_trading():
    """Check why agent isn't trading"""
    with app.app_context():
        print("🔍 Diagnosing No Trading Issue...")
        print("=" * 60)
        
        # Check session
        session = TradingSession.query.filter_by(id=1).first()
        if not session:
            print("❌ No session found!")
            return
            
        print(f"\n📊 Session: {session.session_name}")
        print(f"   Episodes: {session.current_episode}")
        
        # Check recent trades
        recent_trades = Trade.query.filter_by(session_id=1).order_by(Trade.id.desc()).limit(5).all()
        print(f"\n📈 Recent Trades in Session:")
        for trade in recent_trades:
            print(f"   Trade {trade.id}: Entry ${trade.entry_price:.2f}, P&L: ${trade.profit_loss:.2f}")
        
        # Check logs for trading actions
        print("\n📄 Checking Trading Logs:")
        log_files = ['logs/trading_first_1.log', 'logs/trading.log', 'logs/futures_env.log']
        
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"\n   Checking {log_file}:")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Look for recent entries
                    recent_lines = lines[-50:] if len(lines) > 50 else lines
                    
                    action_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
                    rewards = []
                    
                    for line in recent_lines:
                        if 'ACTION:' in line:
                            for action in action_count:
                                if action in line:
                                    action_count[action] += 1
                        if 'reward:' in line or 'Reward:' in line:
                            try:
                                # Extract reward value
                                parts = line.split('reward:')
                                if len(parts) < 2:
                                    parts = line.split('Reward:')
                                if len(parts) >= 2:
                                    reward_str = parts[1].strip().split()[0].replace(',', '')
                                    reward = float(reward_str)
                                    rewards.append(reward)
                            except:
                                pass
                    
                    print(f"     Actions: BUY={action_count['BUY']}, SELL={action_count['SELL']}, HOLD={action_count['HOLD']}")
                    if rewards:
                        print(f"     Recent rewards: min={min(rewards):.2f}, max={max(rewards):.2f}, avg={sum(rewards)/len(rewards):.2f}")
                    
                    # Check for position info
                    position_lines = [line for line in recent_lines if 'position' in line.lower()]
                    if position_lines:
                        print(f"     Recent position info:")
                        for line in position_lines[-3:]:
                            print(f"       {line.strip()}")
        
        print("\n💡 Common Reasons for No Trading:")
        print("1. Agent is being cautious after the big loss")
        print("2. Reward function might be penalizing trades too heavily")
        print("3. Agent might be stuck in a 'HOLD' policy")
        print("4. Position limits might be preventing new trades")
        
        print("\n🛠️  Suggestions:")
        print("1. Check if MAX_POSITION_SIZE is limiting trades")
        print("2. The agent might need more episodes to recover confidence")
        print("3. Consider adjusting reward shaping in futures_env.py")

if __name__ == "__main__":
    diagnose_no_trading()