#!/bin/bash
# Install all project dependencies

echo "Installing all project dependencies..."

# Install all dependencies listed in pyproject.toml
pip install \
    email-validator \
    flask-socketio \
    flask \
    flask-sqlalchemy \
    gunicorn \
    psycopg2-binary \
    werkzeug \
    yfinance \
    python-socketio \
    eventlet \
    ib-insync \
    pandas \
    numpy \
    sqlalchemy \
    ta \
    python-dotenv \
    websocket-client \
    pytz \
    gym \
    matplotlib \
    scipy \
    scikit-learn \
    psutil \
    pytest \
    pytest-cov \
    pytest-mock \
    requests \
    pyarrow \
    socketio

# Install PyTorch (CPU version for quick start, change to GPU if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "✓ All dependencies installed!"
echo ""
echo "Now starting training..."
./start_training.sh