#!/bin/bash
# Check trading activity on remote machine

echo "🔍 Checking Trading Activity"
echo "================================"

# Find the correct PostgreSQL user
echo "Available PostgreSQL users:"
sudo -u postgres psql -c "\du" 2>/dev/null || echo "Need sudo access"

# Try different connection methods
echo -e "\n📊 Checking recent trades (last minute):"
# Method 1: Using trader_user
psql -U trader_user -d reinforcement_trader -c "SELECT COUNT(*) as trades_last_minute FROM trade WHERE created_at > NOW() - INTERVAL '1 minute'" 2>/dev/null

# Method 2: Using sudo postgres
sudo -u postgres psql -d reinforcement_trader -c "SELECT COUNT(*) as trades_last_minute FROM trade WHERE created_at > NOW() - INTERVAL '1 minute'" 2>/dev/null

# Method 3: Using environment variable
if [ ! -z "$DATABASE_URL" ]; then
    psql "$DATABASE_URL" -c "SELECT COUNT(*) as trades_last_minute FROM trade WHERE created_at > NOW() - INTERVAL '1 minute'" 2>/dev/null
fi

echo -e "\n📈 Last 20 trades with details:"
# Show recent trades
psql -U trader_user -d reinforcement_trader -c "
SELECT 
    id,
    TO_CHAR(entry_time, 'HH24:MI:SS') as entry_time,
    TO_CHAR(exit_time, 'HH24:MI:SS') as exit_time,
    entry_price,
    exit_price,
    profit_loss,
    CASE WHEN exit_time IS NULL THEN 'OPEN' ELSE 'CLOSED' END as status
FROM trade 
WHERE session_id = 1 
ORDER BY id DESC 
LIMIT 20
" 2>/dev/null || sudo -u postgres psql -d reinforcement_trader -c "
SELECT 
    id,
    TO_CHAR(entry_time, 'HH24:MI:SS') as entry_time,
    TO_CHAR(exit_time, 'HH24:MI:SS') as exit_time,
    entry_price,
    exit_price,
    profit_loss,
    CASE WHEN exit_time IS NULL THEN 'OPEN' ELSE 'CLOSED' END as status
FROM trade 
WHERE session_id = 1 
ORDER BY id DESC 
LIMIT 20
"

echo -e "\n📊 Trade creation rate (last 10 minutes):"
psql -U trader_user -d reinforcement_trader -c "
SELECT 
    TO_CHAR(created_at, 'HH24:MI') as minute,
    COUNT(*) as trades_created
FROM trade 
WHERE session_id = 1 
    AND created_at > NOW() - INTERVAL '10 minutes'
GROUP BY TO_CHAR(created_at, 'HH24:MI')
ORDER BY minute DESC
" 2>/dev/null || echo "Could not get trade rate"

echo -e "\n💾 Check log file for live activity:"
# Get the latest log entries
tail -n 50 logs/trading_Full_Dataset_Sequential_1.log | grep -E "(TRADE ENTRY|TRADE EXIT|Episode)" | tail -n 20