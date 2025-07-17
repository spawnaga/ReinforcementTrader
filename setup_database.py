#!/usr/bin/env python3
"""
PostgreSQL Database Setup Script
Sets up the database for the Reinforcement Trading System
"""

import os
import sys
import argparse
import psycopg2
from psycopg2 import sql
import subprocess
from pathlib import Path


def check_postgresql_installed():
    """Check if PostgreSQL is installed"""
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ PostgreSQL is installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("✗ PostgreSQL is not installed")
    print("Please install PostgreSQL first. See POSTGRESQL_SETUP.md for instructions.")
    return False


def test_connection(host, port, user, password, database=None):
    """Test database connection"""
    try:
        conn_string = f"host={host} port={port} user={user} password={password}"
        if database:
            conn_string += f" dbname={database}"
        
        conn = psycopg2.connect(conn_string)
        conn.close()
        return True
    except Exception as e:
        return False


def create_database(host, port, admin_user, admin_password, db_name, db_user, db_password):
    """Create database and user"""
    try:
        # Connect to PostgreSQL as admin
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=admin_user,
            password=admin_password,
            dbname='postgres'
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        if cur.fetchone():
            print(f"Database '{db_name}' already exists")
        else:
            # Create database
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            print(f"✓ Created database '{db_name}'")
        
        # Check if user exists
        cur.execute("SELECT 1 FROM pg_user WHERE usename = %s", (db_user,))
        if cur.fetchone():
            print(f"User '{db_user}' already exists")
            # Update password
            cur.execute(sql.SQL("ALTER USER {} WITH PASSWORD %s").format(sql.Identifier(db_user)), (db_password,))
            print(f"✓ Updated password for user '{db_user}'")
        else:
            # Create user
            cur.execute(sql.SQL("CREATE USER {} WITH PASSWORD %s").format(sql.Identifier(db_user)), (db_password,))
            print(f"✓ Created user '{db_user}'")
        
        # Grant privileges
        cur.execute(sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
            sql.Identifier(db_name),
            sql.Identifier(db_user)
        ))
        print(f"✓ Granted all privileges on '{db_name}' to '{db_user}'")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


def create_env_file(host, port, db_name, db_user, db_password):
    """Create or update .env file"""
    env_path = Path('.env')
    
    # Read existing content if file exists
    existing_content = []
    database_url_found = False
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith('DATABASE_URL='):
                    database_url_found = True
                    existing_content.append(f"DATABASE_URL=postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}\n")
                else:
                    existing_content.append(line)
    
    # Add DATABASE_URL if not found
    if not database_url_found:
        if existing_content and not existing_content[-1].endswith('\n'):
            existing_content.append('\n')
        existing_content.append(f"DATABASE_URL=postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}\n")
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(existing_content)
    
    print(f"✓ Updated .env file with DATABASE_URL")


def create_tables(host, port, db_name, db_user, db_password):
    """Create database tables"""
    try:
        # Set environment variable
        os.environ['DATABASE_URL'] = f"postgresql://{db_user}:{db_password}@{host}:{port}/{db_name}"
        
        # Import and create tables
        from app import app, db
        
        with app.app_context():
            db.create_all()
            print("✓ Created all database tables")
        
        return True
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        print("You may need to install required packages: pip install flask flask-sqlalchemy psycopg2-binary")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Setup PostgreSQL database for Reinforcement Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup with default values (localhost)
  python setup_database.py
  
  # Setup with custom database name and user
  python setup_database.py --db-name trading_db --db-user trader
  
  # Setup on remote host
  python setup_database.py --host 192.168.1.100 --admin-user postgres
  
  # Just create the .env file
  python setup_database.py --env-only
        """
    )
    
    # Database connection parameters
    parser.add_argument('--host', default='localhost', help='PostgreSQL host (default: localhost)')
    parser.add_argument('--port', type=int, default=5432, help='PostgreSQL port (default: 5432)')
    
    # Admin credentials (for creating database)
    parser.add_argument('--admin-user', default='postgres', help='PostgreSQL admin user (default: postgres)')
    parser.add_argument('--admin-password', help='PostgreSQL admin password (will prompt if not provided)')
    
    # Database and user to create
    parser.add_argument('--db-name', default='reinforcement_trader', help='Database name to create (default: reinforcement_trader)')
    parser.add_argument('--db-user', default='trader_user', help='Database user to create (default: trader_user)')
    parser.add_argument('--db-password', help='Database user password (will generate if not provided)')
    
    # Actions
    parser.add_argument('--env-only', action='store_true', help='Only create/update .env file')
    parser.add_argument('--skip-tables', action='store_true', help='Skip creating tables')
    
    args = parser.parse_args()
    
    # Check PostgreSQL installation
    if not args.env_only and not check_postgresql_installed():
        return 1
    
    # Get passwords
    if not args.env_only and not args.admin_password:
        import getpass
        args.admin_password = getpass.getpass(f"Enter password for PostgreSQL admin user '{args.admin_user}': ")
    
    if not args.db_password:
        import secrets
        import string
        # Generate secure password
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        args.db_password = ''.join(secrets.choice(alphabet) for _ in range(16))
        print(f"Generated password for '{args.db_user}': {args.db_password}")
    
    # Create/update .env file
    create_env_file(args.host, args.port, args.db_name, args.db_user, args.db_password)
    
    if args.env_only:
        print("\n✓ Setup complete! .env file has been updated.")
        return 0
    
    # Test admin connection
    print(f"\nTesting connection to PostgreSQL as '{args.admin_user}'...")
    if not test_connection(args.host, args.port, args.admin_user, args.admin_password):
        print("✗ Failed to connect to PostgreSQL as admin user")
        print("Please check your admin credentials and PostgreSQL service status")
        return 1
    
    print("✓ Successfully connected to PostgreSQL")
    
    # Create database and user
    print(f"\nCreating database '{args.db_name}' and user '{args.db_user}'...")
    if not create_database(args.host, args.port, args.admin_user, args.admin_password, 
                          args.db_name, args.db_user, args.db_password):
        return 1
    
    # Test new user connection
    print(f"\nTesting connection as '{args.db_user}'...")
    if not test_connection(args.host, args.port, args.db_user, args.db_password, args.db_name):
        print("✗ Failed to connect with new user")
        return 1
    
    print("✓ Successfully connected with new user")
    
    # Create tables
    if not args.skip_tables:
        print("\nCreating database tables...")
        if not create_tables(args.host, args.port, args.db_name, args.db_user, args.db_password):
            print("Warning: Failed to create tables. You may need to run this manually later.")
    
    print("\n" + "="*60)
    print("✓ PostgreSQL setup complete!")
    print("="*60)
    print(f"Database: {args.db_name}")
    print(f"User: {args.db_user}")
    print(f"Host: {args.host}:{args.port}")
    print(f"\nConnection string saved to .env file")
    print("\nYou can now run training with:")
    print("python trading_cli.py --train --algorithm ane_ppo --ticker NQ --episodes 100")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())