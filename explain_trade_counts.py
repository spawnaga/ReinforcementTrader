#!/usr/bin/env python3
"""
Explain the difference between performance trades and system trades
"""
from app import app
from extensions import db
from models import TradingSession, Trade
from sqlalchemy import text

def explain_trade_counts():
    """Show the difference between session trades and total system trades"""
    with app.app_context():
        print("🔍 Understanding Trade Count Differences")
        print("=" * 60)
        
        # Get active session
        active_session = TradingSession.query.filter_by(status='active').first()
        
        if active_session:
            print(f"\n📊 Active Session: {active_session.session_name} (ID: {active_session.id})")
            
            # Count trades for THIS session only
            session_trades = Trade.query.filter_by(session_id=active_session.id).count()
            session_closed_trades = Trade.query.filter_by(
                session_id=active_session.id
            ).filter(Trade.exit_time.isnot(None)).count()
            
            print(f"\n🎯 Performance Section (Current Session Only):")
            print(f"   Total Trades: {session_closed_trades} (closed trades with P&L)")
            print(f"   Open Positions: {session_trades - session_closed_trades}")
            print(f"   Session Total: {session_trades}")
            
        # Count ALL trades in the system
        total_system_trades = Trade.query.count()
        
        print(f"\n💻 System Resources Section (All Time):")
        print(f"   Total Trades: {total_system_trades} (across ALL sessions ever)")
        
        # Show breakdown by session
        print(f"\n📈 Trade Breakdown by Session:")
        session_stats = db.session.execute(text("""
            SELECT 
                ts.id,
                ts.session_name,
                ts.status,
                COUNT(t.id) as trade_count,
                SUM(CASE WHEN t.exit_time IS NOT NULL THEN 1 ELSE 0 END) as closed_trades,
                COALESCE(SUM(t.profit_loss), 0) as total_pnl
            FROM trading_session ts
            LEFT JOIN trade t ON ts.id = t.session_id
            GROUP BY ts.id, ts.session_name, ts.status
            ORDER BY ts.id DESC
            LIMIT 10
        """))
        
        for row in session_stats:
            status_icon = "🟢" if row.status == "active" else "⭕"
            print(f"   {status_icon} Session {row.id} '{row.session_name}': {row.trade_count} trades ({row.closed_trades} closed), P&L: ${row.total_pnl:.2f}")
        
        print(f"\n💡 Explanation:")
        print("1. Performance Section = Trades for CURRENT active session only")
        print("2. System Resources = ALL trades EVER made across ALL sessions")
        print("3. Each training session accumulates its own trades")
        print("4. Old sessions' trades remain in the database")
        print("\nThis is why you see 2 trades in Performance but 26,500+ in System Resources!")

if __name__ == "__main__":
    explain_trade_counts()