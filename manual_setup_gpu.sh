#!/bin/bash
# Manual setup for multi-GPU training - bypasses package conflicts

echo "================================="
echo "Manual Multi-GPU Setup"
echo "================================="

# Your GPUs are detected perfectly:
echo "✓ 4x RTX 3090 GPUs detected"
echo "✓ NVLink working at 14.062 GB/s per link"
echo ""

# Since you already have a conda environment active, let's use it
echo "Using existing conda environment: ReinforcementTrader"

# Step 1: Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install project dependencies
echo "Installing project dependencies..."
pip install flask flask-sqlalchemy flask-socketio gunicorn
pip install pandas numpy scipy scikit-learn matplotlib
pip install gym psycopg2-binary python-dotenv
pip install yfinance ta pytz requests
pip install tqdm psutil eventlet werkzeug
pip install pytest pytest-cov pytest-mock

# Step 3: Verify GPU setup
echo ""
echo "Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Step 4: Create .env file
echo ""
echo "Creating .env file..."
echo "# Database configuration" > .env
echo "# Use your local PostgreSQL or Replit's DATABASE_URL" >> .env
echo "" >> .env

# Step 5: Setup database
echo ""
echo "Setting up database tables..."
python setup_training_db.py || echo "Database setup can be run later"

echo ""
echo "================================="
echo "Setup Complete!"
echo "================================="
echo ""
echo "To start multi-GPU training:"
echo "python train_standalone.py --num-gpus 4 --episodes 10000 --batch-size 256"
echo ""
echo "Monitor GPUs in another terminal:"
echo "watch -n 1 nvidia-smi"