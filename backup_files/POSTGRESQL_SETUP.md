# PostgreSQL Setup Guide for Local Development

## Overview
The trading system has been updated to use PostgreSQL exclusively for better concurrent access during multi-GPU training. SQLite's single-writer limitation was causing issues with your 4x RTX 3090 setup.

## Prerequisites
- PostgreSQL 12 or higher
- Python environment with psycopg2-binary

## Step 1: Install PostgreSQL

### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

### On macOS:
```bash
brew install postgresql
brew services start postgresql
```

### On Windows:
Download and install from https://www.postgresql.org/download/windows/

## Step 2: Create Database and User

1. Access PostgreSQL as the postgres user:
```bash
sudo -u postgres psql
```

2. Create a database and user:
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

## Step 3: Set Environment Variable

Add this to your shell configuration file (.bashrc, .zshrc, etc.):

```bash
export DATABASE_URL="postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader"
```

Then reload your shell:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Step 4: Install Python Dependencies

Make sure you have the PostgreSQL adapter:
```bash
pip install psycopg2-binary
```

## Step 5: Initialize the Database

Run the application once to create all tables:
```bash
python run_local.py
```

## Step 6: Migrate Existing Data (Optional)

If you have existing data in SQLite that you want to keep:

1. Create a migration script `migrate_local_data.py`:

```python
import os
import pandas as pd
from sqlalchemy import create_engine, text

# Source SQLite database
sqlite_url = 'sqlite:///instance/trading_system.db'
# Target PostgreSQL database
postgres_url = os.environ.get('DATABASE_URL')

if not postgres_url:
    print("ERROR: DATABASE_URL environment variable not set!")
    exit(1)

print(f"Migrating from SQLite to PostgreSQL...")

# Create engines
sqlite_engine = create_engine(sqlite_url)
postgres_engine = create_engine(postgres_url)

# Get list of tables
with sqlite_engine.connect() as conn:
    tables_query = text("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name NOT LIKE 'sqlite_%'
    """)
    tables = [row[0] for row in conn.execute(tables_query)]

print(f"Found {len(tables)} tables to migrate: {tables}")

# Migrate each table
for table_name in tables:
    try:
        df = pd.read_sql_table(table_name, sqlite_engine)
        if len(df) > 0:
            df.to_sql(table_name, postgres_engine, if_exists='append', index=False)
            print(f"✓ Migrated {len(df)} rows from table '{table_name}'")
        else:
            print(f"✓ Table '{table_name}' is empty")
    except Exception as e:
        print(f"✗ Error migrating '{table_name}': {e}")

print("Migration complete!")
```

2. Run the migration:
```bash
python migrate_local_data.py
```

## Step 7: Update run_local.py (Temporary Fix)

Until the run_local.py is updated to support PostgreSQL, you can use this modified version:

```python
#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    print("\n🚀 Starting Revolutionary AI Trading System (Local Development)...")
    print("=" * 60)
    
    # Check for DATABASE_URL
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("❌ ERROR: DATABASE_URL environment variable not set!")
        print("\nPlease set it up:")
        print("export DATABASE_URL='postgresql://trader_user:password@localhost:5432/reinforcement_trader'")
        return
    
    print(f"✓ Using PostgreSQL database")
    
    # Start the server
    print("\n🌐 Starting server on http://127.0.0.1:5000")
    print("✓ Using threaded worker for better compatibility")
    print("=" * 60)
    
    print("\n⚡ The application is starting...")
    print("📊 Open your browser to http://127.0.0.1:5000")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run with gunicorn
    cmd = [
        "gunicorn",
        "--bind", "127.0.0.1:5000",
        "--worker-class", "gthread",
        "--workers", "1",
        "--threads", "8",
        "--timeout", "120",
        "--reload",
        "main:app"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Connection Refused
- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check if it's listening on the correct port: `sudo netstat -plnt | grep postgres`

### Authentication Failed
- Verify your DATABASE_URL format
- Check pg_hba.conf for authentication method (should allow md5 or scram-sha-256)

### Permission Denied
- Make sure the user has proper permissions on the database
- Re-run the GRANT command from Step 2

## Benefits of PostgreSQL

1. **True concurrent access** - Multiple processes can write simultaneously
2. **Better performance** - Optimized for multi-GPU training workloads
3. **Advanced features** - JSON support, full-text search, etc.
4. **Production-ready** - Same database for development and production

## Next Steps

After setup, the system will automatically use PostgreSQL with optimized connection pooling (30 connections + 20 overflow) for your multi-GPU training setup.