#!/bin/bash
# Alternative TA-Lib installation using conda-forge

echo "Alternative TA-Lib installation using conda..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Installing TA-Lib from conda-forge..."
    conda install -c conda-forge ta-lib -y
    
    echo "Testing installation..."
    python -c "import talib; print('TA-Lib version:', talib.__version__)"
    
    if [ $? -eq 0 ]; then
        echo "✓ TA-Lib installed successfully via conda!"
    else
        echo "✗ Installation failed"
    fi
else
    echo "Conda not found. Installing using system packages..."
    
    # For Ubuntu 20.04+
    sudo apt-get update
    sudo apt-get install -y libta-lib0 libta-lib0-dev
    
    # Set environment variables for pip
    export TA_INCLUDE_PATH=/usr/include
    export TA_LIBRARY_PATH=/usr/lib
    
    # Install Python wrapper
    pip install numpy
    pip install TA-Lib
    
    echo "Testing installation..."
    python -c "import talib; print('TA-Lib version:', talib.__version__)"
fi