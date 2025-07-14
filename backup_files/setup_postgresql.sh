#!/bin/bash
# PostgreSQL Setup Script for the Trading System

echo "=================================="
echo "PostgreSQL Setup for Trading System"
echo "=================================="

# Check if PostgreSQL is installed
if ! command -v psql &> /dev/null; then
    echo "❌ PostgreSQL is not installed."
    echo ""
    echo "Please install PostgreSQL first:"
    echo "  Ubuntu/Debian: sudo apt install postgresql postgresql-contrib"
    echo "  macOS: brew install postgresql"
    echo "  Windows: Download from https://www.postgresql.org/download/windows/"
    exit 1
fi

echo "✅ PostgreSQL is installed"

# Default values
DB_NAME="reinforcement_trader"
DB_USER="trader_user"
DB_PASS="trading_system_2025"

# Ask for custom values
echo ""
read -p "Database name [$DB_NAME]: " input_db
DB_NAME=${input_db:-$DB_NAME}

read -p "Database user [$DB_USER]: " input_user
DB_USER=${input_user:-$DB_USER}

read -sp "Database password [$DB_PASS]: " input_pass
echo ""
DB_PASS=${input_pass:-$DB_PASS}

# Create the database and user
echo ""
echo "Creating database and user..."

sudo -u postgres psql <<EOF
-- Create database
CREATE DATABASE $DB_NAME;

-- Create user
CREATE USER $DB_USER WITH PASSWORD '$DB_PASS';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;

-- Show result
\l $DB_NAME
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Database setup complete!"
    echo ""
    echo "Add this to your shell configuration file (.bashrc, .zshrc, etc.):"
    echo ""
    echo "export DATABASE_URL=\"postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME\""
    echo ""
    echo "Then reload your shell:"
    echo "  source ~/.bashrc  # or ~/.zshrc"
    echo ""
    echo "Or run this command to set it temporarily:"
    echo "  export DATABASE_URL=\"postgresql://$DB_USER:$DB_PASS@localhost:5432/$DB_NAME\""
else
    echo ""
    echo "❌ Database setup failed. Please check the error messages above."
fi