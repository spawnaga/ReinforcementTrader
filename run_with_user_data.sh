#!/bin/bash
# Script to run training with user's actual data file

echo "Running training with your data file..."
echo "Please replace YOUR_DATA_FILE.csv with your actual file path"
echo

# Command to run with user's data
python train_standalone.py \
    --episodes 10 \
    --ticker NQ \
    --algorithm ANE-PPO \
    --data-file YOUR_DATA_FILE.csv

# Monitor logs for the bug
echo
echo "Monitoring for 11,735 bug..."
tail -f logs/latest/*.log | grep -E "(11[0-9]{3}|STEP REWARD TRACE|Episode [0-9]+ .*Reward)"