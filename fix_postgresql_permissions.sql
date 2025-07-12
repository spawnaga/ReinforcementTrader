-- Fix PostgreSQL permissions for the trading system
-- Run this as the postgres superuser

-- Connect to the database
\c reinforcement_trader

-- Grant all privileges on the public schema
GRANT ALL ON SCHEMA public TO trader_user;

-- Grant all privileges on all tables in public schema
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader_user;

-- Grant all privileges on all sequences in public schema
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader_user;

-- Grant all privileges on all functions in public schema
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO trader_user;

-- Ensure trader_user can create objects in public schema
GRANT CREATE ON SCHEMA public TO trader_user;

-- Make trader_user the owner of the public schema (optional but recommended)
ALTER SCHEMA public OWNER TO trader_user;

-- Verify permissions
\dn+ public