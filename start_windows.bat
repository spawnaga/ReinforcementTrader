@echo off
echo.
echo ================================================
echo Starting Revolutionary AI Trading System...
echo ================================================
echo.

REM Create instance directory if not exists
if not exist "instance" mkdir instance

REM Set database URL to use absolute path
set DATABASE_URL=sqlite:///%cd%\instance\trading_system.db

echo Using database at: %cd%\instance\trading_system.db
echo.

REM Check if gunicorn is installed
python -c "import gunicorn" 2>nul
if errorlevel 1 (
    echo Installing gunicorn...
    pip install gunicorn
    echo.
)

echo Starting server on http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

REM Start gunicorn
gunicorn --bind 127.0.0.1:5000 --reload --worker-class sync --workers 1 --timeout 120 main:app

pause