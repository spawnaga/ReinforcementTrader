#!/bin/bash
# Complete TA-Lib installation script for Ubuntu/Debian

echo "Installing TA-Lib for Ubuntu/Debian..."

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential wget

# Clean up any previous attempts
echo "Cleaning up previous installations..."
sudo rm -rf /usr/local/include/ta-lib
sudo rm -rf /usr/local/lib/libta_lib*
rm -rf /tmp/ta-lib*

# Download and extract TA-Lib
echo "Downloading TA-Lib 0.4.0..."
cd /tmp
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib

# Configure and compile
echo "Configuring TA-Lib..."
./configure --prefix=/usr/local

echo "Compiling TA-Lib (this may take a few minutes)..."
make -j$(nproc)

echo "Installing TA-Lib..."
sudo make install

# Update library cache
echo "Updating library cache..."
sudo ldconfig

# Clean up
cd /
rm -rf /tmp/ta-lib*

echo "TA-Lib C library installation complete!"
echo ""
echo "Now installing Python wrapper..."

# Install Python wrapper
pip install numpy  # Required dependency
pip install TA-Lib

echo ""
echo "Testing installation..."
python -c "import talib; print('TA-Lib version:', talib.__version__)"

if [ $? -eq 0 ]; then
    echo "✓ TA-Lib installed successfully!"
else
    echo "✗ TA-Lib Python wrapper installation failed"
    echo "Try the alternative method in install_talib_alternative.sh"
fi