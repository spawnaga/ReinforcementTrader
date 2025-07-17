#!/usr/bin/env python3
"""Quick training run to verify timestamp fix works throughout training"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use existing modules
from train_standalone import main, parse_args

# Create test arguments
test_args = [
    '--data-file', 'test_data_sample.csv',
    '--episodes', '2',
    '--max-steps', '3',
    '--batch-size', '2',
    '--epochs-per-loop', '1',
    '--training-loops', '1'
]

# Parse arguments with test values
args = parse_args()
for i in range(0, len(test_args), 2):
    setattr(args, test_args[i].replace('--', '').replace('-', '_'), test_args[i+1])

# Run training
print("Starting test training to verify timestamp fix...")
print("This will create logs with historical timestamps (2021-05-31)")
print("-" * 50)

main(args)

print("\n" + "-" * 50)
print("Training complete! Check the log files:")
print("- logs/latest/algorithm.log - Should show 2021-05-31 timestamps")
print("- logs/latest/rewards.log - Should show 2021-05-31 timestamps")
print("- logs/latest/trading.log - Should show 2021-05-31 timestamps")