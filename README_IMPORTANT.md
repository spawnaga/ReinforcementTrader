# IMPORTANT: How to Run This Application

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

## Database Lock Issues

If you see "database is locked" errors, this is because SQLite doesn't handle concurrent access well. For production use, switch to PostgreSQL:

```bash
export DATABASE_URL=postgresql://username:password@localhost/trading_db
```

The application includes retry logic for SQLite, but PostgreSQL is recommended for better performance.