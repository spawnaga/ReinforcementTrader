#!/usr/bin/env python3
"""
Check what's happening in live training
"""
import os
from app import app
from extensions import db
from models import TradingSession, Trade, TrainingMetrics
from sqlalchemy import text
from datetime import datetime, timedelta

def check_live_training():
    """Monitor live training activity"""
    with app.app_context():
        print("🔍 Checking Live Training Activity")
        print("=" * 60)
        
        # Get active session
        active_session = TradingSession.query.filter_by(status='active').first()
        
        if not active_session:
            print("❌ No active session found")
            return
            
        session_id = active_session.id
        print(f"\n📊 Session: {active_session.session_name} (ID: {session_id})")
        
        # Check total trades over time
        print("\n📈 Trade Activity Over Time:")
        
        # Get trades from last 30 minutes in 5-minute intervals
        now = datetime.utcnow()
        for i in range(6):
            end_time = now - timedelta(minutes=i*5)
            start_time = end_time - timedelta(minutes=5)
            
            count_result = db.session.execute(text("""
                SELECT COUNT(*) 
                FROM trade 
                WHERE session_id = :session_id 
                AND created_at BETWEEN :start_time AND :end_time
            """), {
                "session_id": session_id,
                "start_time": start_time,
                "end_time": end_time
            })
            
            count = count_result.scalar()
            print(f"   {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}: {count} new trades")
        
        # Check if trades are actually different
        print("\n📊 Last 10 Unique Trades:")
        recent_trades = Trade.query.filter_by(
            session_id=session_id
        ).order_by(Trade.id.desc()).limit(10).all()
        
        for trade in recent_trades:
            status = "CLOSED" if trade.exit_time else "OPEN"
            print(f"   Trade #{trade.id}: {status} - Entry: ${trade.entry_price:.2f} @ {trade.entry_time}")
        
        # Check open vs closed trades
        open_trades = Trade.query.filter_by(
            session_id=session_id,
            exit_time=None
        ).count()
        
        closed_trades = Trade.query.filter_by(
            session_id=session_id
        ).filter(Trade.exit_time.isnot(None)).count()
        
        print(f"\n📊 Trade Status:")
        print(f"   Open trades: {open_trades}")
        print(f"   Closed trades: {closed_trades}")
        print(f"   Total: {open_trades + closed_trades}")
        
        # Check if training is actually progressing
        recent_metrics = TrainingMetrics.query.filter_by(
            session_id=session_id
        ).order_by(TrainingMetrics.id.desc()).limit(5).all()
        
        print(f"\n🎯 Recent Episode Progress:")
        for metric in recent_metrics:
            print(f"   Episode {metric.episode}: {metric.timestamp}")
        
        # Check for any errors in logs
        log_file = f"logs/trading_{active_session.session_name.replace(' ', '_')}_{session_id}.log"
        if os.path.exists(log_file):
            print(f"\n📄 Recent Log Activity:")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Get last 20 lines
                for line in lines[-20:]:
                    if 'TRADE' in line or 'ERROR' in line:
                        print(f"   {line.strip()}")

if __name__ == "__main__":
    check_live_training()