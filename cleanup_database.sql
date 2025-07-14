-- AI Trading System Database Cleanup Script
-- This will clear all training sessions and trades while keeping market data
-- Run this on your local PostgreSQL database

-- First, show current data counts
SELECT 'BEFORE CLEANUP' as status;
SELECT 'market_data' as table_name, COUNT(*) as count FROM market_data
UNION ALL
SELECT 'trading_session', COUNT(*) FROM trading_session
UNION ALL
SELECT 'trade', COUNT(*) FROM trade
UNION ALL
SELECT 'training_metrics', COUNT(*) FROM training_metrics
UNION ALL
SELECT 'algorithm_config', COUNT(*) FROM algorithm_config;

-- Delete all trades
DELETE FROM trade;

-- Delete all training metrics
DELETE FROM training_metrics;

-- Delete all trading sessions
DELETE FROM trading_session;

-- Delete all algorithm configs
DELETE FROM algorithm_config;

-- Reset sequences if needed (optional)
-- ALTER SEQUENCE trading_session_id_seq RESTART WITH 1;
-- ALTER SEQUENCE trade_id_seq RESTART WITH 1;
-- ALTER SEQUENCE training_metrics_id_seq RESTART WITH 1;
-- ALTER SEQUENCE algorithm_config_id_seq RESTART WITH 1;

-- Verify cleanup
SELECT 'AFTER CLEANUP' as status;
SELECT 'market_data' as table_name, COUNT(*) as count FROM market_data
UNION ALL
SELECT 'trading_session', COUNT(*) FROM trading_session
UNION ALL
SELECT 'trade', COUNT(*) FROM trade
UNION ALL
SELECT 'training_metrics', COUNT(*) FROM training_metrics
UNION ALL
SELECT 'algorithm_config', COUNT(*) FROM algorithm_config;

-- Show summary
SELECT 'Database cleanup complete. Market data preserved.' as message;