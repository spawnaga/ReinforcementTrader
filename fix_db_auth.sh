#!/bin/bash
# Fix PostgreSQL authentication

echo "==================================="
echo "Fix PostgreSQL Authentication"
echo "==================================="
echo ""
echo "Choose an option:"
echo "1) Reset password for existing trader_user"
echo "2) Check current .env file"
echo "3) Test database connection"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Enter NEW password for trader_user:"
        read -s NEW_PASSWORD
        
        # Reset password
        sudo -u postgres psql << EOF
ALTER USER trader_user WITH PASSWORD '$NEW_PASSWORD';
EOF
        
        # Update .env file
        echo "# Database configuration" > .env
        echo "DATABASE_URL=postgresql://trader_user:$NEW_PASSWORD@localhost:5432/reinforcement_trader" >> .env
        
        echo "✓ Password reset complete!"
        echo "✓ .env file updated"
        
        # Test connection
        echo ""
        echo "Testing connection..."
        python -c "
import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    print('✓ Database connection successful!')
    conn.close()
except Exception as e:
    print(f'✗ Connection failed: {e}')
"
        
        # Setup tables if connection works
        echo ""
        echo "Setting up database tables..."
        python setup_training_db.py
        ;;
        
    2)
        echo "Current .env file:"
        if [ -f .env ]; then
            grep DATABASE_URL .env | sed 's/password:[^@]*/password:*****/g'
        else
            echo "No .env file found!"
        fi
        ;;
        
    3)
        echo "Testing database connection..."
        python -c "
import os
from dotenv import load_dotenv
import psycopg2
load_dotenv()
db_url = os.getenv('DATABASE_URL')
if not db_url:
    print('✗ No DATABASE_URL found in .env')
else:
    print(f'Testing: {db_url.replace(db_url.split(\":\")[2].split(\"@\")[0], \"*****\")}')
    try:
        conn = psycopg2.connect(db_url)
        print('✓ Connection successful!')
        conn.close()
    except Exception as e:
        print(f'✗ Connection failed: {e}')
"
        ;;
esac

echo ""
echo "Done!"