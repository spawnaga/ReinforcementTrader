# PostgreSQL Setup Guide

This guide covers installing and configuring PostgreSQL for the Reinforcement Trading System.

## Table of Contents
1. [Installing PostgreSQL on Ubuntu](#installing-postgresql-on-ubuntu)
2. [Installing PostgreSQL on WSL2](#installing-postgresql-on-wsl2)
3. [Quick Setup with Script](#quick-setup-with-script)
4. [Manual Setup](#manual-setup)
5. [Troubleshooting](#troubleshooting)

## Installing PostgreSQL on Ubuntu

### 1. Update System Packages
```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install PostgreSQL
```bash
# Install PostgreSQL and additional contrib package
sudo apt install postgresql postgresql-contrib -y

# Install Python PostgreSQL adapter
pip install psycopg2-binary
```

### 3. Verify Installation
```bash
# Check PostgreSQL version
psql --version

# Check service status
sudo systemctl status postgresql

# Start PostgreSQL if not running
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## Installing PostgreSQL on WSL2

### 1. Install PostgreSQL
```bash
# Update packages
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib -y

# Install Python adapter
pip install psycopg2-binary
```

### 2. Start PostgreSQL Service
In WSL2, you need to manually start PostgreSQL:
```bash
# Start the service
sudo service postgresql start

# Check status
sudo service postgresql status

# To start automatically (add to ~/.bashrc)
echo "sudo service postgresql start" >> ~/.bashrc
```

### 3. Configure PostgreSQL for WSL2
```bash
# Edit PostgreSQL configuration
sudo nano /etc/postgresql/*/main/postgresql.conf

# Find and uncomment/modify:
listen_addresses = 'localhost'
```

## Quick Setup with Script

### 1. Run the Setup Script
```bash
# Basic setup with defaults
python setup_database.py

# Custom setup
python setup_database.py --db-name my_trading_db --db-user my_trader

# Setup with specific admin password
python setup_database.py --admin-password your_admin_pass --db-password your_user_pass
```

### 2. What the Script Does
- Checks if PostgreSQL is installed
- Creates a database named `reinforcement_trader`
- Creates a user named `trader_user`
- Grants all privileges to the user
- Creates/updates the `.env` file
- Creates all necessary tables

## Manual Setup

### 1. Access PostgreSQL
```bash
# Switch to postgres user
sudo -u postgres psql
```

### 2. Create Database and User
```sql
-- Create database
CREATE DATABASE reinforcement_trader;

-- Create user with password
CREATE USER trader_user WITH PASSWORD 'your_secure_password';

-- Grant all privileges
GRANT ALL PRIVILEGES ON DATABASE reinforcement_trader TO trader_user;

-- Exit
\q
```

### 3. Configure Environment
Create or update `.env` file in project root:
```bash
DATABASE_URL=postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader
```

### 4. Create Tables
```bash
# Set environment variable
export DATABASE_URL=postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader

# Run Python to create tables
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

## Verify Setup

### 1. Test Database Connection
```bash
# Connect to database
psql -h localhost -U trader_user -d reinforcement_trader

# List tables (after creation)
\dt

# Exit
\q
```

### 2. Test Python Connection
```python
import psycopg2

conn = psycopg2.connect(
    "postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader"
)
cur = conn.cursor()
cur.execute("SELECT version();")
print(cur.fetchone())
cur.close()
conn.close()
```

## Common PostgreSQL Commands

### Service Management
```bash
# Start/Stop/Restart
sudo systemctl start postgresql
sudo systemctl stop postgresql
sudo systemctl restart postgresql

# For WSL2
sudo service postgresql start
sudo service postgresql stop
sudo service postgresql restart
```

### Database Operations
```sql
-- List all databases
\l

-- Connect to database
\c database_name

-- List all tables
\dt

-- Describe table
\d table_name

-- List all users
\du

-- Drop database (careful!)
DROP DATABASE database_name;

-- Drop user
DROP USER username;
```

## Troubleshooting

### 1. PostgreSQL Not Starting
```bash
# Check logs
sudo journalctl -xe | grep postgres

# Check port usage
sudo lsof -i :5432

# Reset PostgreSQL
sudo systemctl stop postgresql
sudo systemctl start postgresql
```

### 2. Authentication Failed
```bash
# Edit pg_hba.conf
sudo nano /etc/postgresql/*/main/pg_hba.conf

# Change authentication method for local connections to md5:
# local   all             all                                     md5
# host    all             all             127.0.0.1/32            md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### 3. WSL2 Specific Issues
```bash
# If PostgreSQL won't start
sudo rm /var/run/postgresql/.s.PGSQL.5432.lock
sudo service postgresql start

# Permission issues
sudo chown postgres:postgres /var/run/postgresql
sudo chmod 2775 /var/run/postgresql
```

### 4. Python Connection Issues
```bash
# Install required packages
pip install psycopg2-binary sqlalchemy flask-sqlalchemy

# If psycopg2 fails to install
sudo apt install libpq-dev python3-dev
pip install psycopg2
```

## Security Best Practices

1. **Use Strong Passwords**: Generate secure passwords for database users
2. **Limit Connections**: Configure pg_hba.conf to only allow necessary connections
3. **Regular Backups**: Use pg_dump for regular backups
4. **Update Regularly**: Keep PostgreSQL updated with security patches

## Backup and Restore

### Backup Database
```bash
pg_dump -h localhost -U trader_user -d reinforcement_trader > backup.sql
```

### Restore Database
```bash
psql -h localhost -U trader_user -d reinforcement_trader < backup.sql
```

## Next Steps

After setting up PostgreSQL:

1. Run the training system:
   ```bash
   python trading_cli.py --train --algorithm ane_ppo --ticker NQ --episodes 100
   ```

2. Monitor the database:
   ```bash
   psql -h localhost -U trader_user -d reinforcement_trader -c "SELECT COUNT(*) FROM trading_session;"
   ```

3. Start the web application (optional):
   ```bash
   python run.py
   ```