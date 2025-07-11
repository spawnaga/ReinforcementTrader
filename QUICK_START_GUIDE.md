# Quick Start Guide - Revolutionary AI Trading System

## The Database Error Fix

When you see "sqlite3.OperationalError: unable to open database file", it means the database path wasn't set correctly. I've fixed this issue now.

## How to Start the Application

### Option 1 - For Windows/PyCharm Users:
```bash
# Double-click or run in terminal:
start_windows.bat
```

### Option 2 - For Mac/Linux/PyCharm:
```bash
python START_HERE.py
```

### Option 3 - Manual Start:
```bash
# First, install gunicorn if needed
pip install gunicorn

# Then run:
gunicorn --bind 127.0.0.1:5000 --reload main:app
```

## Important Notes

1. **Never use `python -m flask run`** - This causes WebSocket errors
2. The database is now stored in the `instance/` folder
3. The application runs on: http://127.0.0.1:5000

## If You Still Get Errors

1. **Permission Error**: Make sure you have write permissions in the project folder
2. **Port Already in Use**: Change the port in the command (e.g., use 5001 instead of 5000)
3. **Missing Dependencies**: Run `pip install -r requirements.txt` if available

## PyCharm Configuration

To run in PyCharm:
1. Right-click on `START_HERE.py`
2. Select "Run 'START_HERE'"
3. Or create a new Run Configuration with the script path set to `START_HERE.py`

The application will automatically:
- Create the instance directory if needed
- Set up the database with the correct path
- Install gunicorn if it's missing
- Start the server with proper WebSocket support