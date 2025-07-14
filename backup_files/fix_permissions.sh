#!/bin/bash
# Fix PostgreSQL permissions for the trading system

echo "Fixing PostgreSQL permissions..."

# Run the SQL commands as postgres user
sudo -u postgres psql reinforcement_trader <<EOF
-- Grant all privileges on the public schema
GRANT ALL ON SCHEMA public TO trader_user;

-- Grant create privilege
GRANT CREATE ON SCHEMA public TO trader_user;

-- Grant all privileges on future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO trader_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO trader_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO trader_user;

-- Show current permissions
\dn+ public
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Permissions fixed successfully!"
    echo "You can now run: python run_local.py"
else
    echo ""
    echo "❌ Failed to fix permissions. Please check the error messages above."
fi