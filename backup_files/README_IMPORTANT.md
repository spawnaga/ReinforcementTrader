# IMPORTANT: How to Run This Application

## ⚠️ PostgreSQL Setup Required (NEW)

The system now requires PostgreSQL. SQLite has been completely removed for better multi-GPU performance.

### Easiest Setup (Using .env file):

A `.env` file has been created with your database credentials. Just run:
```bash
python run_local.py
```

The application will automatically load the DATABASE_URL from the .env file.

### Manual PostgreSQL Setup:
```bash
# 1. Install PostgreSQL (if needed)
sudo apt install postgresql postgresql-contrib  # Ubuntu/Debian
# OR: brew install postgresql  # macOS

# 2. Run setup script
chmod +x setup_postgresql.sh
./setup_postgresql.sh

# 3. Set environment variable (or use .env file)
export DATABASE_URL="postgresql://trader_user:your_password@localhost:5432/reinforcement_trader"

# 4. Migrate existing data (optional)
python migrate_local_data.py
```

See `POSTGRESQL_SETUP.md` and `LOCAL_DEVELOPMENT.md` for detailed instructions.

## DO NOT USE `flask run` or `python -m flask run`

Flask's development server has a known bug with WebSocket connections that causes the "write() before start_response" error.

## Instead, use ONE of these methods:

### Method 1: Use the provided runner script (RECOMMENDED)
```bash
python run_local.py
```

### Method 2: Use Gunicorn directly
```bash
gunicorn --bind 127.0.0.1:5000 --reload main:app
```

### Method 3: If you don't have Gunicorn installed
```bash
pip install gunicorn
python run_local.py
```

## The application will NOT work properly with `flask run`!

The WebSocket errors you're seeing are because Flask's development server cannot handle WebSocket upgrades properly. This is a known limitation, not a bug in our code.

## Database Requirements

The system now uses PostgreSQL exclusively. SQLite support has been removed to enable proper multi-GPU concurrent training without database locking issues.