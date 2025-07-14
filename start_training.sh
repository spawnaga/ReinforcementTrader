#!/bin/bash
# Load environment variables and start training

echo "Loading environment variables..."
export $(grep -v '^#' .env | xargs)

echo "Starting training with realistic constraints..."
python start_working_training.py