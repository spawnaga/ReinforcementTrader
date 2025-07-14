#!/usr/bin/env python3
"""
Check if trades are actually closing instantly or if it's a display issue
"""
import psycopg2
import os
from datetime import datetime

# Get database connection
db_url = os.environ.get('DATABASE_URL', 'postgresql://trader_user:trader_password@localhost:5432/reinforcement_trader')

try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    print("🔍 Analyzing Trade Timing Patterns")
    print("=" * 60)
    
    # Get detailed timing for recent trades
    cursor.execute("""
        SELECT 
            id,
            entry_time,
            exit_time,
            exit_time - entry_time as duration,
            entry_price,
            exit_price,
            profit_loss
        FROM trade 
        WHERE session_id = 1 
            AND exit_time IS NOT NULL
        ORDER BY id DESC 
        LIMIT 20
    """)
    
    trades = cursor.fetchall()
    
    print("\n📊 Recent Trade Durations:")
    print(f"{'ID':<8} {'Entry Time':<20} {'Exit Time':<20} {'Duration':<15} {'P&L':<10}")
    print("-" * 85)
    
    instant_trades = 0
    for trade in trades:
        trade_id, entry, exit, duration, entry_p, exit_p, pnl = trade
        if duration.total_seconds() == 0:
            instant_trades += 1
        print(f"{trade_id:<8} {str(entry):<20} {str(exit):<20} {str(duration):<15} ${pnl:<10.2f}")
    
    print(f"\n⚡ Instant trades (0 duration): {instant_trades}/{len(trades)}")
    
    # Check unique timestamps
    cursor.execute("""
        SELECT COUNT(DISTINCT entry_time) as unique_entries,
               COUNT(DISTINCT exit_time) as unique_exits,
               COUNT(*) as total_trades
        FROM trade 
        WHERE session_id = 1 
            AND exit_time IS NOT NULL
        ORDER BY id DESC 
        LIMIT 1000
    """)
    
    result = cursor.fetchone()
    if result:
        unique_entries, unique_exits, total = result
        print(f"\n📈 Timestamp Analysis (last 1000 trades):")
        print(f"   Unique entry times: {unique_entries}")
        print(f"   Unique exit times: {unique_exits}")
        print(f"   Total trades: {total}")
        print(f"   Trades per entry time: {total/unique_entries:.1f}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
    print("Try running with: sudo -u postgres python3 check_trade_timing.py")