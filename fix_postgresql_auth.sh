#!/bin/bash

# Fix PostgreSQL authentication for local development
echo "=== PostgreSQL Authentication Fix ==="
echo "This script will configure PostgreSQL for local authentication"
echo ""

# 1. Check if PostgreSQL is running
echo "1. Checking PostgreSQL status..."
if sudo service postgresql status | grep -q "online"; then
    echo "✓ PostgreSQL is running"
else
    echo "✗ PostgreSQL is not running. Starting it now..."
    sudo service postgresql start
fi

# 2. Set password for postgres user
echo ""
echo "2. Setting password for postgres user..."
echo "You'll be prompted to enter a new password for the 'postgres' user"
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"

# 3. Configure authentication
echo ""
echo "3. Configuring PostgreSQL authentication..."
PG_VERSION=$(ls /etc/postgresql/)
PG_CONFIG="/etc/postgresql/$PG_VERSION/main/pg_hba.conf"

echo "Backing up current configuration..."
sudo cp $PG_CONFIG ${PG_CONFIG}.backup

echo "Updating authentication methods..."
# Change local authentication to md5
sudo sed -i 's/local   all             postgres                                peer/local   all             postgres                                md5/' $PG_CONFIG
sudo sed -i 's/local   all             all                                     peer/local   all             all                                     md5/' $PG_CONFIG

# 4. Restart PostgreSQL
echo ""
echo "4. Restarting PostgreSQL..."
sudo service postgresql restart

echo ""
echo "=== Configuration Complete ==="
echo "PostgreSQL is now configured for password authentication."
echo ""
echo "Default credentials set:"
echo "  Username: postgres"
echo "  Password: postgres"
echo ""
echo "You can now run: python setup_database.py --admin-password postgres"