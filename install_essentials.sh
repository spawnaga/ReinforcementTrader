#!/bin/bash
# Install only the essential dependencies for the trading system

echo "Installing essential dependencies for trading system..."

# Core Flask dependencies
pip install flask flask-sqlalchemy flask-socketio python-dotenv

# Database
pip install psycopg2-binary sqlalchemy

# Data processing
pip install pandas numpy pytz

# Machine learning (CPU version for quick start)
pip install torch gym scipy scikit-learn

# Trading specific
pip install yfinance matplotlib

# Additional utilities
pip install psutil eventlet websocket-client

echo ""
echo "✓ Essential dependencies installed!"
echo ""
echo "Starting training now..."
./start_training.sh