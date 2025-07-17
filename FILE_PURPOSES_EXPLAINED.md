# AI Trading System - What Each File Does

## Core Application Files (Keep These)

### app.py
- Sets up Flask application with database and middleware
- Configures PostgreSQL connection
- Initializes database tables
- **Status**: KEEP - Essential core file

### routes.py  
- Defines all API endpoints (/api/sessions, /api/trades, etc.)
- Handles HTTP requests and responses
- **Status**: KEEP - Essential for API functionality

### models.py
- Database table definitions (MarketData, TradingSession, Trade, etc.)
- SQLAlchemy ORM models
- **Status**: KEEP - Essential for database

### run.py
- Production-ready application runner
- Starts Flask with SocketIO support
- Shows API endpoints on startup
- **Status**: KEEP - Main entry point

### websocket_handler.py
- Real-time WebSocket communication
- Broadcasts performance metrics, trade updates
- **Status**: KEEP - Essential for real-time features

## Trading Core (Keep All)

### trading_engine.py
- Main trading logic and algorithms
- Manages training sessions
- Coordinates all trading operations
- **Status**: KEEP - Heart of the system

### trading_cli.py
- Command-line interface with all parameters
- Controls GPU, episodes, algorithms, etc.
- **Status**: KEEP - User interface

### data_manager.py
- Loads and processes market data
- Handles CSV files and database data
- **Status**: KEEP - Data pipeline

### risk_manager.py
- Position sizing and risk controls
- Stop-loss and take-profit logic
- **Status**: KEEP - Safety features

### technical_indicators.py
- RSI, MACD, Bollinger Bands calculations
- Manual implementations (no TA-Lib needed)
- **Status**: KEEP - Market analysis

### futures_contracts.py
- NQ and ES contract specifications
- Tick values, margins, trading hours
- **Status**: KEEP - Contract details

### trading_logger.py
- Detailed trading activity logging
- Records entries, exits, profits
- **Status**: KEEP - Trading history

## Duplicates to Delete

### main.py
- Simple Flask runner (4 lines)
- **Status**: DELETE - Use run.py instead

### monitor.py, gpu_monitor.py, check_training_status.py
- Basic monitoring scripts
- **Status**: DELETE - Use training_monitor.py instead

### clean_db.py, cleanup_database.sql, fix_postgresql_permissions.sql
- Database cleanup scripts
- **Status**: DELETE - Use cleanup_local_database.py instead

### demo_cli.py, demo_gpu_training.py
- Example commands (already in README.md)
- **Status**: DELETE - Documentation duplicates

### check_status.ps1, get_session_details.ps1
- Windows PowerShell scripts
- **Status**: DELETE - Use Python tools instead

### futures_env_realistic.py (in root)
- Trading environment file
- **Status**: DELETE - Already in gym_futures/envs/

## Files to Merge

### extensions.py (14 lines)
- Just creates db and socketio objects
- **Status**: MERGE into app.py

### trading_config.py
- Trading-specific configuration class
- **Status**: MERGE into config.py as a section

## Supporting Files (Keep)

### config.py
- Application configuration
- Database settings, API keys
- **Status**: KEEP (merge trading_config.py into it)

### db_utils.py
- Database retry decorators
- Error handling utilities
- **Status**: KEEP - Useful utilities

### gpu_data_loader.py
- GPU-accelerated data loading
- CUDA optimizations
- **Status**: KEEP - Performance feature

### ib_integration.py
- Interactive Brokers API connection
- Live trading capabilities
- **Status**: KEEP - Broker integration

### run_training.py
- Training process coordinator
- Sets up devices and data
- **Status**: KEEP - Training helper

### training_monitor.py
- Comprehensive monitoring tool
- Shows all metrics in terminal
- **Status**: KEEP - Best monitoring tool

### cleanup_local_database.py
- Interactive database cleanup
- Preserves market data
- **Status**: KEEP - Maintenance tool

## Summary

**Current**: 44 files
**After cleanup**: 29 files

**Delete**: 12 duplicate/unnecessary files
**Merge**: 2 files into existing ones
**Keep**: 30 essential files

This cleanup will:
- Remove 34% of files
- Eliminate all duplicates
- Keep all functionality
- Make the project easier to navigate