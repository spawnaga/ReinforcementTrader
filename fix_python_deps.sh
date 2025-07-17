#!/bin/bash
# Fix Python dependency conflicts on Ubuntu

echo "Fixing Python dependency conflicts..."

# Option 1: Try to fix the held packages
sudo apt-mark unhold python3 python3-venv 2>/dev/null
sudo apt --fix-broken install -y

# Option 2: Install specific Python version
sudo apt install -y python3.12 python3.12-dev python3.12-venv

# Option 3: If venv still fails, create virtual environment using built-in venv
if ! python3 -m venv --help &>/dev/null; then
    echo "Installing venv via pip..."
    python3 -m pip install --user virtualenv
    echo "You can create virtual environment with: python3 -m virtualenv venv"
else
    echo "Python venv is working!"
fi

echo "Fix complete. Now you can:"
echo "1. Create venv: python3 -m venv venv"
echo "2. Activate it: source venv/bin/activate"
echo "3. Continue setup: pip install -r requirements.txt"