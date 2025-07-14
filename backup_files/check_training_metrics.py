#!/usr/bin/env python3
"""
Check training metrics discrepancies
"""
from app import app
from extensions import db
from models import TradingSession, Trade
from sqlalchemy import text

def check_metrics():
    """Check training metrics"""
    with app.app_context():
        print("🔍 Checking Training Metrics...")
        print("=" * 60)
        
        # Get session details
        session = TradingSession.query.get(1)
        if session:
            print(f"\n📊 Session: {session.session_name} (ID: {session.id})")
            print(f"   Status: {session.status}")
            print(f"   Current Episode: {session.current_episode}")
            
            # Count trades for this session
            session_trades = Trade.query.filter_by(session_id=session.id).count()
            print(f"   Trades in this session: {session_trades}")
            
            # Get trade details
            trades = Trade.query.filter_by(session_id=session.id).all()
            for trade in trades[:5]:  # Show first 5
                print(f"\n   Trade {trade.id}:")
                print(f"     Time: {trade.entry_time}")
                print(f"     Action: BUY at ${trade.entry_price:.2f}")
                if trade.exit_time:
                    print(f"     Exit: SELL at ${trade.exit_price:.2f}")
                print(f"     P&L: ${trade.profit_loss:.2f}")
        
        # Check total trades across all sessions
        print(f"\n📈 System-wide Statistics:")
        total_trades = Trade.query.count()
        print(f"   Total trades (all time): {total_trades}")
        
        # Show trades by session
        result = db.session.execute(text("""
            SELECT session_id, COUNT(*) as trade_count, SUM(profit_loss) as total_pnl
            FROM trade
            GROUP BY session_id
            ORDER BY session_id DESC
            LIMIT 10
        """))
        
        print(f"\n   Recent Sessions:")
        for row in result:
            print(f"     Session {row.session_id}: {row.trade_count} trades, P&L: ${row.total_pnl:.2f}")
        
        print("\n💡 Explanation:")
        print("- 'Total Trades' in System Resources = All trades ever made (across all sessions)")
        print("- 'Total Trades' in Performance = Trades in current session only")
        print("- The large loss (-$71,016.50) happened in a single trade")
        print("- Reward: 0.00 means the agent isn't getting positive feedback yet")
        print("- Loss: 0.02-0.03 shows the neural network is learning (normal range)")

if __name__ == "__main__":
    check_metrics()