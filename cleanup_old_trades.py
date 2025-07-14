#!/usr/bin/env python3
"""
Clean up old trades and sessions (optional)
"""
from app import app
from extensions import db
from models import TradingSession, Trade, TrainingMetrics
from sqlalchemy import text

def cleanup_old_data():
    """Optionally clean up old trading data"""
    with app.app_context():
        print("🧹 Database Cleanup Tool")
        print("=" * 60)
        
        # Show current counts
        total_sessions = TradingSession.query.count()
        total_trades = Trade.query.count()
        total_metrics = TrainingMetrics.query.count()
        
        print(f"\nCurrent Database Status:")
        print(f"  Sessions: {total_sessions}")
        print(f"  Trades: {total_trades}")
        print(f"  Training Metrics: {total_metrics}")
        
        # Get active session
        active_session = TradingSession.query.filter_by(status='active').first()
        if active_session:
            print(f"\n⚠️  Active Session: {active_session.session_name} (ID: {active_session.id})")
            print("   This session will be preserved!")
        
        response = input("\nDo you want to clean up old data? (y/N): ")
        
        if response.lower() == 'y':
            preserve_id = active_session.id if active_session else None
            
            # Delete old data
            if preserve_id:
                print(f"\nPreserving active session {preserve_id}...")
                # Delete trades not from active session
                deleted_trades = Trade.query.filter(Trade.session_id != preserve_id).delete()
                # Delete metrics not from active session
                deleted_metrics = TrainingMetrics.query.filter(TrainingMetrics.session_id != preserve_id).delete()
                # Delete sessions except active one
                deleted_sessions = TradingSession.query.filter(TradingSession.id != preserve_id).delete()
            else:
                print("\nCleaning all data...")
                deleted_trades = Trade.query.delete()
                deleted_metrics = TrainingMetrics.query.delete()
                deleted_sessions = TradingSession.query.delete()
            
            db.session.commit()
            
            print(f"\n✅ Cleanup Complete:")
            print(f"   Deleted {deleted_sessions} sessions")
            print(f"   Deleted {deleted_trades} trades")
            print(f"   Deleted {deleted_metrics} metrics")
            
            # Show new counts
            new_sessions = TradingSession.query.count()
            new_trades = Trade.query.count()
            
            print(f"\nNew Database Status:")
            print(f"  Sessions: {new_sessions}")
            print(f"  Trades: {new_trades}")
        else:
            print("\n❌ Cleanup cancelled")

if __name__ == "__main__":
    cleanup_old_data()