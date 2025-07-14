#!/usr/bin/env python3
"""
Check why reward is always 0.0
"""
import os
from app import app
from extensions import db
from models import TradingSession, TrainingMetrics
from sqlalchemy import text

def check_reward_issue():
    """Investigate the reward calculation issue"""
    with app.app_context():
        print("🔍 Investigating Reward Issue")
        print("=" * 60)
        
        # Get active session
        active_session = TradingSession.query.filter_by(status='active').first()
        
        if not active_session:
            print("❌ No active session found")
            return
            
        print(f"\n📊 Session: {active_session.session_name} (ID: {active_session.id})")
        
        # Get last 10 training metrics
        recent_metrics = TrainingMetrics.query.filter_by(
            session_id=active_session.id
        ).order_by(TrainingMetrics.id.desc()).limit(10).all()
        
        print(f"\n📈 Last 10 Episode Rewards:")
        for metric in reversed(recent_metrics):
            print(f"   Episode {metric.episode}: Reward = {metric.reward:.4f}, Loss = {metric.loss:.4f}")
        
        # Check log files for reward details
        log_file = f"logs/trading_{active_session.session_name.replace(' ', '_')}_{active_session.id}.log"
        
        if os.path.exists(log_file):
            print(f"\n📄 Checking {log_file} for reward calculations...")
            
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
                # Look for reward-related entries
                reward_lines = []
                for line in lines[-200:]:  # Last 200 lines
                    if 'REWARD' in line or 'reward' in line or 'net_profit' in line:
                        reward_lines.append(line.strip())
                
                if reward_lines:
                    print("\nRecent Reward Calculations:")
                    for line in reward_lines[-10:]:  # Show last 10
                        print(f"   {line}")
        
        print("\n💡 Common Reasons for 0.0 Reward:")
        print("1. The agent is mostly HOLDING positions (no reward until trade closes)")
        print("2. Trades are opening but not closing within the episode")
        print("3. The reward function only gives rewards on COMPLETED trades")
        print("4. Early in training, the agent may be too cautious")
        
        # Check if trades are being closed
        recent_trades = db.session.execute(text("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN exit_time IS NOT NULL THEN 1 ELSE 0 END) as closed,
                SUM(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END) as open
            FROM trade 
            WHERE session_id = :session_id
        """), {"session_id": active_session.id})
        
        result = recent_trades.fetchone()
        if result:
            print(f"\n📊 Trade Status:")
            print(f"   Total trades: {result.total}")
            print(f"   Closed trades: {result.closed} ({result.closed/result.total*100:.1f}%)")
            print(f"   Open trades: {result.open}")
            
            if result.open > result.closed:
                print("\n⚠️  Many trades are still OPEN - this explains the 0.0 rewards!")
                print("    Rewards are only calculated when positions are CLOSED")

if __name__ == "__main__":
    check_reward_issue()