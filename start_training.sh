#!/bin/bash
# Start training script for NQ futures with ANE-PPO

echo "Starting ANE-PPO training for NQ futures..."
echo "Dataset: 4.3 million rows from ./data/processed/NQ_train_processed.csv"
echo "Indicators: Cyclical time features + Technical indicators"
echo ""

# Clear Python cache to avoid import issues
echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# Run training
python trading_cli.py --train \
    --algorithm ane_ppo \
    --ticker NQ \
    --data-source ./data/processed/NQ_train_processed.csv \
    --device gpu \
    --num-gpus 1 \
    --use-transformer \
    --use-genetic \
    --episodes 100 \
    --indicators sin_time cos_time sin_weekday cos_weekday sin_hour cos_hour SMA EMA RSI MACD BB ATR