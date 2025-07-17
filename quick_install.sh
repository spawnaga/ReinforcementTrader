#!/bin/bash
# Quick install of remaining dependencies based on your errors

echo "Installing remaining dependencies..."

# Install the core missing dependencies
pip install psycopg2-binary flask-socketio pytz

# Then run training
echo ""
echo "Dependencies installed. Starting training..."
./start_training.sh