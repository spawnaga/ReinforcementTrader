#!/bin/bash
# Start ANE-PPO training with NQ futures data

echo "Starting ANE-PPO training for NQ futures..."
echo "Dataset: 4.3 million rows from ./data/processed/NQ_train_processed.csv"
echo "Indicators: Cyclical time features + Technical indicators"
echo ""

# Clear Python cache to avoid import issues
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Step 1: Setup database tables (if needed)
echo "Setting up database tables..."
python setup_database.py 2>/dev/null || echo "Database tables already exist or not using database"

echo ""

# Step 2: Start training with standalone script (avoids circular imports)
echo "Starting training with standalone script..."
python train_standalone.py