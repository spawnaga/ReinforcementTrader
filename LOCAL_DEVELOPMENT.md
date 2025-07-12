# Local Development Guide

## Running the Application Locally

### Avoiding WebSocket Errors

The Flask development server has known issues with WebSocket connections. To run the application locally without errors, use one of these methods:

### Method 1: Use the Local Runner Script (Recommended)

```bash
python run_local.py
```

This script runs the application using Gunicorn, which properly handles WebSocket connections.

### Method 2: Use Gunicorn Directly

```bash
gunicorn --bind 127.0.0.1:5000 --worker-class sync --workers 1 --threads 4 --timeout 120 --reload --log-level debug main:app
```

### Method 3: Install and Configure for Flask Development Server

If you must use Flask's development server, disable WebSocket transport:

1. Set environment variable:
```bash
export FLASK_SOCKETIO_TRANSPORTS=polling
```

2. Run Flask:
```bash
flask run
```

## Database Considerations

### SQLite Locking Issues

SQLite has limitations with concurrent access. The application includes retry logic to handle database locks, but for better performance consider:

1. **PostgreSQL for Local Development**: Install PostgreSQL and set the DATABASE_URL environment variable:
```bash
export DATABASE_URL=postgresql://username:password@localhost/trading_db
```

2. **Single Worker Mode**: When using SQLite, always run with a single worker to minimize locking issues.

## Environment Variables

Create a `.env` file for local development:

```env
FLASK_ENV=development
FLASK_DEBUG=1
SESSION_SECRET=your-secret-key-here
DATABASE_URL=sqlite:///trading_system.db
LOG_LEVEL=DEBUG
```

## Troubleshooting

### "AssertionError: write() before start_response"
This is a WebSocket compatibility issue with Flask's development server. Use Method 1 or 2 above.

### "sqlite3.OperationalError: database is locked"
This happens when multiple threads access SQLite simultaneously. The application includes retry logic, but using PostgreSQL eliminates this issue.

### "attempt to write a readonly database"
This is a file permission issue. Run the fix script:
```bash
python fix_db_permissions.py
```

Or manually fix permissions:
```bash
# Fix database file permissions
chmod 664 instance/trading_system.db
chmod 664 instance/trading_system.db-wal
chmod 664 instance/trading_system.db-shm

# Fix directory permissions
chmod 775 instance/

# Change ownership if needed
chown $USER:$USER instance/trading_system.db*
```

For WSL users: Move the database to the Linux filesystem (not /mnt/c/) or adjust Windows permissions.

### Socket.IO Connection Errors
If you see repeated connection/disconnection messages, ensure you're using Gunicorn instead of Flask's development server.

## GPU Support

To use GPU acceleration locally:

1. Install CUDA toolkit (11.7 or higher)
2. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

The application will automatically detect and use GPU if available.