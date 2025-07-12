#!/usr/bin/env python3
"""
Migration script to move data from SQLite to PostgreSQL
For local development environments
"""
import os
import sys
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

def main():
    print("=" * 60)
    print("SQLite to PostgreSQL Migration Tool")
    print("=" * 60)
    
    # Check for PostgreSQL URL
    postgres_url = os.environ.get('DATABASE_URL')
    if not postgres_url:
        print("❌ ERROR: DATABASE_URL environment variable not set!")
        print("\nPlease set up PostgreSQL first. See POSTGRESQL_SETUP.md for instructions.")
        sys.exit(1)
    
    # Find SQLite database
    sqlite_paths = [
        'instance/trading_system.db',
        Path.home() / 'PycharmProjects' / 'ReinforcementTrader' / 'instance' / 'trading_system.db'
    ]
    
    sqlite_path = None
    for path in sqlite_paths:
        if os.path.exists(path):
            sqlite_path = path
            break
    
    if not sqlite_path:
        print("❌ No SQLite database found to migrate.")
        print("Checked locations:")
        for path in sqlite_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print(f"✓ Found SQLite database at: {sqlite_path}")
    print(f"✓ Migrating to PostgreSQL: {postgres_url.split('@')[-1]}")
    
    # Create engines
    sqlite_url = f'sqlite:///{sqlite_path}'
    sqlite_engine = create_engine(sqlite_url)
    postgres_engine = create_engine(postgres_url)
    
    # Get list of tables
    with sqlite_engine.connect() as conn:
        tables_query = text("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [row[0] for row in conn.execute(tables_query)]
    
    if not tables:
        print("❌ No tables found in SQLite database.")
        sys.exit(1)
    
    print(f"\n📊 Found {len(tables)} tables to migrate:")
    for table in tables:
        print(f"  - {table}")
    
    # Ask for confirmation
    response = input("\n⚠️  This will ADD data to your PostgreSQL database. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Migration cancelled.")
        sys.exit(0)
    
    print("\n🔄 Starting migration...")
    
    total_rows = 0
    successful_tables = 0
    
    for table_name in tables:
        try:
            # Read data from SQLite
            df = pd.read_sql_table(table_name, sqlite_engine)
            row_count = len(df)
            
            if row_count > 0:
                # Write to PostgreSQL
                df.to_sql(table_name, postgres_engine, if_exists='append', index=False)
                print(f"✅ {table_name}: Migrated {row_count} rows")
                total_rows += row_count
            else:
                print(f"⚡ {table_name}: Empty table (skipped)")
            
            successful_tables += 1
            
        except Exception as e:
            print(f"❌ {table_name}: Error - {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"✨ Migration Complete!")
    print(f"   - Tables migrated: {successful_tables}/{len(tables)}")
    print(f"   - Total rows: {total_rows}")
    print("=" * 60)
    
    if successful_tables < len(tables):
        print("\n⚠️  Some tables failed to migrate. Check the errors above.")
    else:
        print("\n✅ All data successfully migrated to PostgreSQL!")
        print("\nYou can now run the application with:")
        print("  python run_local.py")

if __name__ == "__main__":
    main()