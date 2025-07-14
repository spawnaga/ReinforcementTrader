#!/usr/bin/env python3
"""
Check training logs and database for detailed information
"""
from app import app
from extensions import db
from models import TradingSession, Trade, TrainingMetrics
from sqlalchemy import text, desc
import pandas as pd

def check_training_details():
    """Check detailed training information"""
    with app.app_context():
        print("🔍 Checking Training Details...")
        print("=" * 80)
        
        # Get active sessions
        active_sessions = TradingSession.query.filter_by(status='active').all()
        print(f"\n📊 Active Sessions: {len(active_sessions)}")
        
        for session in active_sessions:
            print(f"\n🚀 Session ID: {session.id} - {session.session_name}")
            print(f"   Status: {session.status}")
            print(f"   Algorithm: {session.algorithm_type}")
            print(f"   Started: {session.start_time}")
            print(f"   Current Episode: {session.current_episode}/{session.total_episodes}")
            print(f"   Parameters: {session.parameters}")
            
            # Check for trades
            trades = Trade.query.filter_by(session_id=session.id).order_by(desc(Trade.entry_time)).limit(5).all()
            print(f"\n   Recent Trades ({len(trades)} total):")
            for trade in trades:
                print(f"   - {trade.position_type.upper()} @ {trade.entry_price} "
                      f"(P/L: ${trade.profit_loss:.2f}) - Episode {trade.episode_number}")
            
            # Check training metrics
            metrics = TrainingMetrics.query.filter_by(session_id=session.id).order_by(desc(TrainingMetrics.episode)).limit(5).all()
            print(f"\n   Recent Training Metrics ({len(metrics)} total):")
            for metric in metrics:
                print(f"   - Episode {metric.episode}: Reward={metric.reward:.2f}, "
                      f"Loss={metric.loss:.4f if metric.loss else 0}, LR={metric.learning_rate}")
        
        # Check market data
        print("\n📈 Market Data Check:")
        result = db.session.execute(text("SELECT COUNT(*) FROM market_data"))
        count = result.scalar()
        print(f"   Total market data records: {count:,}")
        
        # Check latest market data
        result = db.session.execute(text("""
            SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest 
            FROM market_data
        """))
        row = result.fetchone()
        if row:
            print(f"   Date range: {row.oldest} to {row.newest}")
        
        # Check if there are any errors in logs
        print("\n🔍 Checking for recent errors...")
        # Check if log files exist
        import os
        log_files = ['logs/trading_positions.log', 'logs/trading_errors.log', 'logs/trading_debug.log']
        for log_file in log_files:
            if os.path.exists(log_file):
                print(f"\n   Checking {log_file}:")
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        # Get last 10 lines
                        recent_lines = lines[-10:] if len(lines) > 10 else lines
                        for line in recent_lines:
                            if 'ERROR' in line or 'WARNING' in line:
                                print(f"   ⚠️  {line.strip()}")
                except Exception as e:
                    print(f"   Could not read {log_file}: {e}")

if __name__ == "__main__":
    check_training_details()