#!/usr/bin/env python3
"""
Direct database monitor - works without Flask dependencies
"""
import psycopg2
import os
import time
from datetime import datetime
import sys

def get_db_connection():
    """Get direct PostgreSQL connection"""
    # Try different methods to get database URL
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        # Try reading from .env file
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    if line.startswith('DATABASE_URL='):
                        db_url = line.split('=', 1)[1].strip().strip('"\'')
                        break
    
    if not db_url:
        # Default local connection
        db_url = "postgresql://trader_user:trader_password@localhost:5432/reinforcement_trader"
    
    return psycopg2.connect(db_url)

def clear_screen():
    """Clear screen cross-platform"""
    os.system('clear' if os.name != 'nt' else 'cls')

def monitor_training():
    """Monitor training directly from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        while True:
            clear_screen()
            print("=" * 80)
            print(f"{'🚀 Direct Database Monitor':^80}")
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
            print("=" * 80)
            
            # Get active sessions
            cursor.execute("""
                SELECT id, session_name, status, algorithm, total_episodes, 
                       created_at, updated_at
                FROM trading_session 
                WHERE status = 'active'
                ORDER BY id DESC
                LIMIT 5
            """)
            sessions = cursor.fetchall()
            
            if not sessions:
                print("\n❌ No active training sessions")
                print("\nStart training with: python3 run_local.py")
            else:
                for session in sessions:
                    session_id, name, status, algo, total_ep, created, updated = session
                    
                    print(f"\n📊 Session: {name} (ID: {session_id})")
                    print(f"   Status: {status}")
                    print(f"   Algorithm: {algo}")
                    print(f"   Started: {created}")
                    
                    # Get latest metrics
                    cursor.execute("""
                        SELECT episode, reward, loss 
                        FROM training_metrics 
                        WHERE session_id = %s 
                        ORDER BY episode DESC 
                        LIMIT 1
                    """, (session_id,))
                    
                    metric = cursor.fetchone()
                    if metric:
                        episode, reward, loss = metric
                        progress = (episode / total_ep) * 100 if total_ep > 0 else 0
                        
                        print(f"\n🎯 Training Progress:")
                        print(f"   Episodes: {episode}/{total_ep}")
                        
                        # Progress bar
                        bar_length = 50
                        filled = int(bar_length * progress / 100)
                        bar = "█" * filled + "░" * (bar_length - filled)
                        print(f"   [{bar}] {progress:.1f}%")
                        print(f"   Latest Reward: {reward:.2f}")
                        print(f"   Latest Loss: {loss:.4f}")
                    
                    # Get performance metrics
                    cursor.execute("""
                        SELECT COUNT(*) as total_trades,
                               SUM(profit_loss) as total_profit,
                               SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
                        FROM trade 
                        WHERE session_id = %s AND exit_time IS NOT NULL
                    """, (session_id,))
                    
                    trade_stats = cursor.fetchone()
                    if trade_stats and trade_stats[0] > 0:
                        total_trades, total_profit, wins = trade_stats
                        win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
                        
                        print(f"\n💰 Performance:")
                        print(f"   Total Trades: {total_trades}")
                        print(f"   Total Profit: ${total_profit:,.2f}" if total_profit else "   Total Profit: $0.00")
                        print(f"   Win Rate: {win_rate:.1f}%")
                    
                    # Get recent trades
                    cursor.execute("""
                        SELECT id, entry_price, exit_price, profit_loss 
                        FROM trade 
                        WHERE session_id = %s AND exit_time IS NOT NULL
                        ORDER BY id DESC 
                        LIMIT 5
                    """, (session_id,))
                    
                    trades = cursor.fetchall()
                    if trades:
                        print(f"\n📊 Recent Trades:")
                        for trade in trades:
                            trade_id, entry, exit, pnl = trade
                            print(f"   #{trade_id}: ${entry:.2f} → ${exit:.2f} P&L: ${pnl:.2f}")
            
            print("\n" + "=" * 80)
            print("Press Ctrl+C to quit | Updates every 5 seconds")
            
            conn.commit()
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure PostgreSQL is accessible and DATABASE_URL is set")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    print("Starting direct database monitor...")
    print("This monitor doesn't require Flask or other web dependencies")
    monitor_training()