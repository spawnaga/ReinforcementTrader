#!/bin/bash
# Complete the TA-Lib installation that's already in progress

echo "Completing TA-Lib installation..."

# Continue from where we left off
cd /tmp/ta-lib

# Compile (this will take a few minutes)
echo "Compiling TA-Lib..."
make -j$(nproc)

# Install
echo "Installing TA-Lib..."
sudo make install

# Update library cache
echo "Updating library cache..."
sudo ldconfig

# Install Python wrapper
echo "Installing Python TA-Lib wrapper..."
pip install numpy  # Ensure numpy is installed first
pip install TA-Lib

# Test installation
echo ""
echo "Testing TA-Lib installation..."
python -c "import talib; print('✓ TA-Lib version:', talib.__version__)" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ TA-Lib installed successfully!"
    echo ""
    echo "You can now run your training with full TA-Lib support:"
    echo "./start_training.sh"
else
    echo "✗ Python wrapper installation failed"
    echo "But don't worry - your code already has manual implementations!"
    echo "You can still run training without TA-Lib."
fi