#!/bin/bash
# Multi-GPU Setup Script for Ubuntu with RTX 3090s
# This script automates the setup process for the reinforcement trader

set -e  # Exit on error

echo "======================================"
echo "Multi-GPU Reinforcement Trader Setup"
echo "======================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only"
    exit 1
fi

# Step 1: Check GPU availability
print_status "Checking GPU configuration..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Check NVLink
echo -e "\nChecking NVLink status:"
nvidia-smi nvlink -s || print_warning "NVLink status could not be determined"

# Step 2: Install system dependencies
print_status "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev build-essential libpq-dev postgresql postgresql-contrib

# Step 3: Create Python virtual environment
print_status "Creating Python virtual environment..."
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Step 4: Upgrade pip and install PyTorch
print_status "Installing PyTorch with CUDA support..."
pip install --upgrade pip

# Detect CUDA version
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
echo "Detected CUDA version: $CUDA_VERSION"

# Install PyTorch based on CUDA version
if [[ "$CUDA_VERSION" == "12.1" ]] || [[ "$CUDA_VERSION" == "12.2" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_warning "CUDA $CUDA_VERSION not directly supported. Installing CPU-only PyTorch."
    print_warning "Please manually install PyTorch with your CUDA version from https://pytorch.org"
    pip install torch torchvision torchaudio
fi

# Step 5: Install project dependencies
print_status "Installing project dependencies..."
pip install -r requirements.txt 2>/dev/null || {
    print_warning "requirements.txt not found or failed. Installing essential packages..."
    pip install flask flask-sqlalchemy flask-socketio gunicorn
    pip install pandas numpy scipy scikit-learn matplotlib
    pip install gym psycopg2-binary python-dotenv
    pip install yfinance ta pytz requests
    pip install tqdm psutil eventlet werkzeug
}

# Step 6: Test GPU detection in Python
print_status "Testing GPU detection in PyTorch..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
else:
    print("No CUDA GPUs detected!")
EOF

# Step 7: Setup PostgreSQL
print_status "Setting up PostgreSQL database..."
echo "Please enter a password for the PostgreSQL trader_user:"
read -s DB_PASSWORD

# Create database and user
sudo -u postgres psql << EOF 2>/dev/null || print_warning "Database may already exist"
CREATE USER trader_user WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE reinforcement_trader OWNER trader_user;
GRANT ALL PRIVILEGES ON DATABASE reinforcement_trader TO trader_user;
EOF

# Step 8: Create .env file
print_status "Creating .env configuration file..."
if [ -f ".env" ]; then
    print_warning ".env file already exists. Backing up to .env.backup"
    cp .env .env.backup
fi

cat > .env << EOF
# Database configuration
DATABASE_URL=postgresql://trader_user:$DB_PASSWORD@localhost:5432/reinforcement_trader

# Add other environment variables here as needed
# SESSION_SECRET=your_session_secret_here
EOF

# Step 9: Initialize database tables
print_status "Initializing database tables..."
python setup_training_db.py || print_error "Failed to setup database tables"

# Step 10: Create helper scripts
print_status "Creating helper scripts..."

# Create GPU monitoring script
cat > monitor_gpus.sh << 'EOF'
#!/bin/bash
# Monitor GPU usage during training
watch -n 1 'nvidia-smi; echo ""; nvidia-smi nvlink -s'
EOF
chmod +x monitor_gpus.sh

# Create training launcher
cat > train_multi_gpu.sh << 'EOF'
#!/bin/bash
# Launch multi-GPU training with optimal settings

# Activate virtual environment
source venv/bin/activate

# Set environment variables for optimal performance
export NCCL_P2P_LEVEL=NVL  # Use NVLink for peer-to-peer
export OMP_NUM_THREADS=8   # Adjust based on CPU cores

# Default parameters (can be overridden by command line args)
NUM_GPUS=${NUM_GPUS:-4}
EPISODES=${EPISODES:-10000}
BATCH_SIZE=${BATCH_SIZE:-256}
ALGORITHM=${ALGORITHM:-ANE-PPO}

echo "Starting training with $NUM_GPUS GPUs..."
echo "Episodes: $EPISODES, Batch Size: $BATCH_SIZE, Algorithm: $ALGORITHM"

# Run training
python train_standalone.py \
    --num-gpus $NUM_GPUS \
    --episodes $EPISODES \
    --batch-size $BATCH_SIZE \
    --algorithm $ALGORITHM \
    "$@"  # Pass any additional arguments
EOF
chmod +x train_multi_gpu.sh

# Create quick test script
cat > test_training.sh << 'EOF'
#!/bin/bash
# Quick test to ensure everything is working

source venv/bin/activate

echo "Running quick training test (100 episodes)..."
python train_standalone.py \
    --num-gpus 4 \
    --episodes 100 \
    --algorithm ANE-PPO \
    --ticker NQ

echo "Test complete! Check logs/latest/ for results."
EOF
chmod +x test_training.sh

print_status "Setup complete!"

echo -e "\n${GREEN}======================================"
echo "Setup Summary:"
echo "======================================${NC}"
echo "✓ Found $GPU_COUNT GPU(s)"
echo "✓ Python virtual environment created"
echo "✓ PyTorch installed with CUDA support"
echo "✓ PostgreSQL database configured"
echo "✓ Helper scripts created"

echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Load your data: python prepare_training_data.py --data-file /path/to/data.csv"
echo "3. Run a test: ./test_training.sh"
echo "4. Start full training: ./train_multi_gpu.sh"
echo "5. Monitor GPUs: ./monitor_gpus.sh (in another terminal)"

echo -e "\n${YELLOW}Useful Commands:${NC}"
echo "- Check logs: tail -f logs/latest/training.log"
echo "- Monitor training: python training_monitor.py"
echo "- Custom training: python train_standalone.py --num-gpus 4 --episodes 10000"

echo -e "\n${GREEN}Happy Training!${NC}"