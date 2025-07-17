#!/bin/bash
# Install all required dependencies for the trading system

echo "Installing all required dependencies..."

# Core dependencies
pip install flask flask-sqlalchemy flask-socketio
pip install python-dotenv psycopg2-binary
pip install sqlalchemy werkzeug

# Data processing
pip install numpy pandas pyarrow
pip install yfinance

# Machine learning
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install gym scikit-learn scipy

# Trading specific
pip install ib-insync
pip install ta  # Technical analysis library (alternative to TA-Lib)

# Additional utilities
pip install matplotlib requests pytz
pip install eventlet websocket-client
pip install psutil gunicorn

# Testing (optional)
pip install pytest pytest-cov pytest-mock

echo ""
echo "✓ All dependencies installed!"
echo ""
echo "Now you can run your training:"
echo "./start_training.sh"