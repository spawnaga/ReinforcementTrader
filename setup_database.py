#!/usr/bin/env python3
"""Setup database with proper permissions"""

import os
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# Parse DATABASE_URL
DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in .env file")
    exit(1)

# Parse the URL
url = urlparse(DATABASE_URL)
db_name = url.path[1:]  # Remove leading '/'
db_user = url.username
db_password = url.password
db_host = url.hostname
db_port = url.port or 5432

print(f"Connecting to database '{db_name}' as user '{db_user}'...")

try:
    # Connect as the user
    conn = psycopg2.connect(
        dbname=db_name,
        user=db_user,
        password=db_password,
        host=db_host,
        port=db_port
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # Grant permissions on public schema
    print("Granting permissions on public schema...")
    cur.execute(f"GRANT CREATE ON SCHEMA public TO {db_user};")
    cur.execute(f"GRANT ALL ON SCHEMA public TO {db_user};")
    print("✓ Permissions granted")
    
    # Create tables
    print("\nCreating tables...")
    from app import app, db
    with app.app_context():
        db.create_all()
        print("✓ All tables created successfully")
    
    cur.close()
    conn.close()
    
except psycopg2.errors.InsufficientPrivilege:
    print("\nERROR: Insufficient privileges. You need to run these commands as a PostgreSQL superuser:")
    print(f"\n1. Connect as postgres user:")
    print(f"   sudo -u postgres psql -d {db_name}")
    print(f"\n2. Run these commands:")
    print(f"   GRANT CREATE ON SCHEMA public TO {db_user};")
    print(f"   GRANT ALL ON SCHEMA public TO {db_user};")
    print(f"   \\q")
    print(f"\n3. Then run this script again")
    
except Exception as e:
    print(f"ERROR: {e}")