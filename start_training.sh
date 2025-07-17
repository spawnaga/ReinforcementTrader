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

# Step 2: Setup PostgreSQL training schemas (optional)
echo "Setting up PostgreSQL training schemas..."
python setup_training_db.py 2>/dev/null || echo "PostgreSQL tracking schemas not created (optional)"

# Step 3: Create necessary directories
mkdir -p logs models database

echo ""
echo "Starting training with professional logging..."
echo "• tqdm progress bars in console"
echo "• Detailed logs in ./logs/latest/"
echo "• PostgreSQL tracking (if available)"
echo ""

# Step 4: Start training with standalone script (avoids circular imports)
python train_standalone.py