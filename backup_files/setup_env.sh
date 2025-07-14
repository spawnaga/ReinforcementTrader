#!/bin/bash
# Setup environment variables for the trading system

echo "=== Trading System Environment Setup ==="
echo

# Check if .env file exists
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    export $(grep -v '^#' .env | xargs)
    echo "✓ Environment loaded from .env"
else
    echo "⚠️  No .env file found"
fi

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo
    echo "❌ DATABASE_URL is not set!"
    echo
    echo "Please set it using one of these methods:"
    echo
    echo "1. Create a .env file with:"
    echo "   DATABASE_URL=postgresql://username:password@host:port/database"
    echo
    echo "2. Export it in your shell:"
    echo "   export DATABASE_URL=postgresql://username:password@host:port/database"
    echo
    echo "3. If using local PostgreSQL:"
    echo "   export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/trading_db"
    echo
    exit 1
else
    echo "✓ DATABASE_URL is set"
    # Show connection info (hide password)
    echo "  Connection: ${DATABASE_URL//:*@/:****@}"
fi

# Check other important variables
echo
echo "Checking other environment variables:"

if [ -z "$SESSION_SECRET" ]; then
    echo "⚠️  SESSION_SECRET not set (using default)"
else
    echo "✓ SESSION_SECRET is set"
fi

if [ -z "$IB_HOST" ]; then
    echo "⚠️  IB_HOST not set (will use default 127.0.0.1)"
else
    echo "✓ IB_HOST is set to: $IB_HOST"
fi

echo
echo "=== Setup Complete ==="
echo
echo "Now you can run:"
echo "  python reset_training.py"
echo "  python check_training_status.py"
echo "  python start_working_training.py"