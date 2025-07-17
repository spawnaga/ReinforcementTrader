#!/usr/bin/env python3
"""Test script to show where debug messages appear"""

import os
import sys
import subprocess
import time

print("=" * 60)
print("DEBUG MESSAGE LOCATION TEST")
print("=" * 60)

# Check if we're in the right directory
if not os.path.exists("train_standalone.py"):
    print("ERROR: Please run this from your ReinforcementTrader directory")
    sys.exit(1)

print("\n1. When you run training, debug messages appear in TWO places:")
print("   a) Your terminal (console output)")
print("   b) Log files in logs/<timestamp>/ directory")

print("\n2. Running a test training session...")
print("   Command: python train_standalone.py --episodes 3 --ticker NQ --algorithm ANE-PPO")
print("\n" + "-" * 60)

# Run training for just 3 episodes
try:
    # Start the training process
    process = subprocess.Popen(
        ["python", "train_standalone.py", "--episodes", "3", "--ticker", "NQ", "--algorithm", "ANE-PPO"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print("\n=== CONSOLE OUTPUT (where debug messages appear) ===\n")
    
    # Read output line by line
    debug_found = False
    for line in process.stdout:
        print(line, end='')
        
        # Check for our debug messages
        if any(msg in line for msg in ["GET_REWARD", "STEP REWARD", "NO TRADE", "REWARD SCALING", "HUGE HOLD"]):
            debug_found = True
            print(f"\n>>> DEBUG MESSAGE FOUND: {line.strip()}\n")
    
    process.wait()
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    process.terminate()
except Exception as e:
    print(f"\nError running training: {e}")

print("\n" + "-" * 60)

# Check log files
print("\n3. Checking log files for debug messages...")

# Find the most recent log directory
log_dirs = sorted([d for d in os.listdir("logs") if d.startswith("2025")])
if log_dirs:
    latest_dir = log_dirs[-1]
    log_path = f"logs/{latest_dir}"
    print(f"   Latest log directory: {log_path}")
    
    # Check each log file
    for log_file in ["algorithm.log", "rewards.log", "trading.log"]:
        file_path = f"{log_path}/{log_file}"
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                debug_msgs = ["GET_REWARD", "STEP REWARD", "NO TRADE", "REWARD SCALING", "HUGE HOLD"]
                found_msgs = [msg for msg in debug_msgs if msg in content]
                if found_msgs:
                    print(f"\n   Found debug messages in {log_file}: {', '.join(found_msgs)}")
                    # Show first few occurrences
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if any(msg in line for msg in debug_msgs):
                            print(f"      Line {i+1}: {line[:100]}...")
                            if i > 5:  # Show only first few
                                break

print("\n" + "=" * 60)
print("SUMMARY:")
print("- Debug messages appear in your terminal when training runs")
print("- They're also saved in log files in logs/<timestamp>/")
print("- Messages only appear when specific conditions are met:")
print("  * Rewards > 100")
print("  * Episode < 25 with 0 trades and step > 190")
print("  * Other suspicious reward patterns")
print("=" * 60)