# Multi-GPU Setup Guide for Ubuntu with 4x RTX 3090

## Prerequisites
- Ubuntu 20.04/22.04 with NVIDIA drivers installed
- CUDA 11.8 or 12.1 (for RTX 3090 support)
- Python 3.9+ 
- PostgreSQL 14+
- Git

## Quick Setup Script

Run this one-liner to clone and setup:
```bash
git clone https://github.com/YOUR_USERNAME/reinforcement-trader.git && cd reinforcement-trader && chmod +x setup_multi_gpu.sh && ./setup_multi_gpu.sh
```

## Manual Setup Steps

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/reinforcement-trader.git
cd reinforcement-trader
```

### 2. Verify GPU Setup
```bash
# Check NVIDIA drivers
nvidia-smi

# Check NVLink status
nvidia-smi nvlink -s

# Should show 4 GPUs with NVLink connections
```

### 3. Install System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Install Python dependencies
sudo apt install -y python3-pip python3-venv python3-dev

# Install CUDA toolkit (if not already installed)
# For CUDA 12.1:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-1

# Install additional libraries
sudo apt install -y build-essential libpq-dev
```

### 4. Setup Python Environment
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 5. Configure PostgreSQL
```bash
# Create database and user
sudo -u postgres psql << EOF
CREATE USER trader_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE reinforcement_trader OWNER trader_user;
GRANT ALL PRIVILEGES ON DATABASE reinforcement_trader TO trader_user;
EOF

# Update .env file with your database credentials
echo "DATABASE_URL=postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader" > .env
```

### 6. Initialize Database
```bash
# Setup database tables
python setup_training_db.py

# Load your data (if you have local data files)
python prepare_training_data.py --data-file /path/to/your/data.csv
```

### 7. Test GPU Detection
```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
"
```

## Multi-GPU Training Configuration

### Option 1: Use All 4 GPUs (Recommended)
```bash
# Train with all 4 GPUs using data parallelism
python train_standalone.py \
    --num-gpus 4 \
    --episodes 10000 \
    --algorithm ANE-PPO \
    --ticker NQ
```

### Option 2: Specific GPU Selection
```bash
# Use specific GPUs (e.g., GPU 0 and 2)
python train_standalone.py \
    --gpu-ids 0,2 \
    --episodes 10000 \
    --algorithm ANE-PPO
```

### Option 3: Advanced Multi-GPU with Custom Batch Sizes
```bash
# Larger batch sizes for multi-GPU efficiency
python train_standalone.py \
    --num-gpus 4 \
    --batch-size 256 \
    --episodes 10000 \
    --training-loops 100 \
    --epochs-per-loop 10
```

## Performance Optimization Tips

### 1. NVLink Optimization
With NVLink, you get faster GPU-to-GPU communication:
```bash
# Enable NVLink monitoring during training
export NCCL_P2P_LEVEL=NVL  # Use NVLink for peer-to-peer
export NCCL_DEBUG=INFO     # See communication details
```

### 2. Memory Management
Each RTX 3090 has 24GB VRAM. Optimize usage:
```bash
# Clear GPU memory before training
python -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU memory during training
watch -n 1 nvidia-smi
```

### 3. Batch Size Scaling
For 4 GPUs, scale batch size appropriately:
- Single GPU: batch_size = 32
- 4 GPUs: batch_size = 128-256

### 4. Data Loading Optimization
```bash
# Use faster data loading with multiple workers
export OMP_NUM_THREADS=8  # Adjust based on CPU cores
python train_standalone.py --num-workers 8 --num-gpus 4
```

## Monitoring Training

### Real-time GPU Monitoring
```bash
# In a separate terminal
python gpu_monitor.py
```

### Training Progress
```bash
# Monitor logs
tail -f logs/latest/training.log

# Check PostgreSQL metrics
python -c "
from training_tracker import check_learning_progress
check_learning_progress('latest')  # or specific session_id
"
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
python train_standalone.py --batch-size 64 --num-gpus 4

# Or use gradient accumulation
python train_standalone.py --gradient-accumulation-steps 4
```

### NVLink Not Working
```bash
# Check NVLink topology
nvidia-smi topo -m

# Should show NV# connections between GPUs
```

### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U trader_user -d reinforcement_trader -h localhost
```

## Multi-Node Training (Advanced)
If you want to use multiple machines:
```bash
# On master node
python train_standalone.py \
    --distributed \
    --world-size 2 \
    --rank 0 \
    --master-addr 192.168.1.100 \
    --master-port 29500

# On worker node
python train_standalone.py \
    --distributed \
    --world-size 2 \
    --rank 1 \
    --master-addr 192.168.1.100 \
    --master-port 29500
```

## Expected Performance
With 4x RTX 3090 NVLinked:
- Training speed: ~4-8x faster than single GPU
- Batch processing: Can handle 4x larger batches
- Memory: 96GB total VRAM available
- NVLink bandwidth: 600 GB/s between GPUs

## Next Steps
1. Start with a test run: `python train_standalone.py --episodes 100 --num-gpus 4`
2. Monitor GPU utilization to ensure all GPUs are being used
3. Adjust hyperparameters based on multi-GPU performance
4. Use TensorBoard for visualization: `tensorboard --logdir logs/`