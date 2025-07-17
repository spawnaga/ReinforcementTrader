#!/usr/bin/env python3
"""Test if timestamp fix works in rewards logging"""

import sys
import subprocess
import time

print("Testing timestamp fix in rewards logging...")
print("Starting training with 10 episodes to check timestamps...")
print("-" * 60)

# Run training for a short test
cmd = [
    sys.executable,
    "train_standalone.py",
    "--episodes", "1",
    "--max-steps", "20",
    "--num-gpus", "0"  # CPU only for quick test
]

# Start the training
proc = subprocess.Popen(cmd)

# Let it run for a bit
time.sleep(5)

# Check the rewards log
print("\nChecking rewards.log for timestamps...")
try:
    with open('logs/latest/rewards.log', 'r') as f:
        lines = f.readlines()[-10:]  # Last 10 lines
        
    print("\nLast 10 lines of rewards.log:")
    for line in lines:
        print(line.strip())
        # Check if timestamp looks like historical data (2021) vs current (2025)
        if "2025-" in line:
            print("⚠️  WARNING: Still using current timestamps!")
        elif "2021-" in line or "2008-" in line:
            print("✓ SUCCESS: Using historical timestamps!")
            
except FileNotFoundError:
    print("No rewards.log found yet")

# Terminate the process
proc.terminate()
proc.wait()

print("\n" + "-" * 60)
print("Test complete. Check the output above for timestamp verification.")