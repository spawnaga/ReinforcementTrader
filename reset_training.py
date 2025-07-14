#!/usr/bin/env python3
"""
Reset training sessions and start fresh
"""
from app import app
from extensions import db
from models import TradingSession, Trade, TrainingMetrics
from sqlalchemy import text

def reset_all_sessions():
    """Reset all training sessions"""
    with app.app_context():
        print("🔄 Resetting Training Sessions...")
        print("=" * 60)
        
        # Get current session count
        session_count = TradingSession.query.count()
        trade_count = Trade.query.count()
        
        print(f"Current state:")
        print(f"  Sessions: {session_count}")
        print(f"  Trades: {trade_count}")
        
        # Stop all active sessions
        active_sessions = TradingSession.query.filter_by(status='active').all()
        for session in active_sessions:
            session.status = 'stopped'
            print(f"  Stopped session {session.id}: {session.session_name}")
        
        db.session.commit()
        
        # Optional: Clear ALL sessions and trades
        response = input("\nDo you want to clear ALL sessions and trades? (y/N): ")
        if response.lower() == 'y':
            print("\nClearing all data...")
            
            # Delete all training metrics
            TrainingMetrics.query.delete()
            
            # Delete all trades
            Trade.query.delete()
            
            # Delete all sessions
            TradingSession.query.delete()
            
            db.session.commit()
            print("✅ All sessions and trades cleared!")
            
            # Reset auto-increment counters
            db.session.execute(text("ALTER SEQUENCE trading_session_id_seq RESTART WITH 1"))
            db.session.execute(text("ALTER SEQUENCE trade_id_seq RESTART WITH 1"))
            db.session.commit()
            print("✅ ID counters reset!")
        
        print("\n🎯 Ready to start fresh!")
        print("\nStart a new training session with:")
        print('curl -X POST http://127.0.0.1:5000/api/start_training \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"session_name": "Fresh Start", "algorithm_type": "ANE_PPO", "total_episodes": 100}\'')

if __name__ == "__main__":
    reset_all_sessions()