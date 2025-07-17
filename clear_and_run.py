#!/usr/bin/env python3
"""Clear Python cache and run training"""

import os
import sys
import shutil

# Clear __pycache__ directories
print("Clearing Python cache...")
for root, dirs, files in os.walk('.'):
    if '__pycache__' in dirs:
        pycache_path = os.path.join(root, '__pycache__')
        print(f"Removing {pycache_path}")
        shutil.rmtree(pycache_path)

# Clear .pyc files
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.pyc'):
            pyc_path = os.path.join(root, file)
            print(f"Removing {pyc_path}")
            os.remove(pyc_path)

print("✓ Cache cleared")

# Now run the training command
print("\nStarting training...")
os.system("""python trading_cli.py --train \
    --algorithm ane_ppo \
    --ticker NQ \
    --data-source ./data/processed/NQ_train_processed.csv \
    --device gpu \
    --num-gpus 1 \
    --use-transformer \
    --use-genetic \
    --episodes 100 \
    --indicators sin_time cos_time sin_weekday cos_weekday sin_hour cos_hour SMA EMA RSI MACD BB ATR""")