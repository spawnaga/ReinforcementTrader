#!/usr/bin/env python3
"""
Reset the training system - clear old sessions and trades
"""
import os
from sqlalchemy import create_engine, text
from datetime import datetime

def reset_training():
    """Reset training by clearing old data"""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ DATABASE_URL not set")
        return
        
    print("=== Training System Reset ===")
    print(f"Time: {datetime.now()}")
    print("=" * 50)
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            # Get current status
            result = conn.execute(text("SELECT COUNT(*) FROM trading_session"))
            session_count = result.scalar()
            
            result = conn.execute(text("SELECT COUNT(*) FROM trade"))
            trade_count = result.scalar()
            
            print(f"\nCurrent state:")
            print(f"  - Sessions: {session_count}")
            print(f"  - Trades: {trade_count}")
            
            # Ask for confirmation
            response = input("\nDo you want to reset? This will:\n"
                           "1. Mark all sessions as 'stopped'\n"
                           "2. Clear all trades\n"
                           "3. Clear all training metrics\n"
                           "Type 'yes' to confirm: ")
            
            if response.lower() != 'yes':
                print("\nReset cancelled.")
                return
            
            # Start transaction
            trans = conn.begin()
            try:
                # Mark all sessions as stopped
                conn.execute(text("""
                    UPDATE trading_session 
                    SET status = 'stopped', 
                        end_time = CURRENT_TIMESTAMP 
                    WHERE status = 'active'
                """))
                
                # Clear all trades
                conn.execute(text("DELETE FROM trade"))
                
                # Clear all training metrics
                conn.execute(text("DELETE FROM training_metrics"))
                
                # Reset session statistics
                conn.execute(text("""
                    UPDATE trading_session 
                    SET current_episode = 0,
                        total_profit = 0.0,
                        total_trades = 0,
                        win_rate = 0.0,
                        sharpe_ratio = 0.0,
                        max_drawdown = 0.0
                """))
                
                trans.commit()
                print("\n✅ Reset complete!")
                
                # Verify
                result = conn.execute(text("SELECT COUNT(*) FROM trade"))
                trade_count = result.scalar()
                
                result = conn.execute(text("SELECT COUNT(*) FROM trading_session WHERE status='active'"))
                active_count = result.scalar()
                
                print(f"\nNew state:")
                print(f"  - Active sessions: {active_count}")
                print(f"  - Total trades: {trade_count}")
                
            except Exception as e:
                trans.rollback()
                print(f"\n❌ Reset failed: {e}")
                
    except Exception as e:
        print(f"\n❌ Database error: {e}")

if __name__ == "__main__":
    reset_training()