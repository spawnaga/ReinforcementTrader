#!/usr/bin/env python3
"""
Simple script to start training with realistic constraints
Run this instead of run_local.py to prevent instant trading exploitation
"""

import subprocess
import sys
import os

def main():
    print("\n🚀 Starting Realistic Trading System")
    print("=" * 60)
    print("This version includes realistic trading constraints:")
    print("✓ Minimum holding period: 10 time steps")
    print("✓ Maximum trades per episode: 5")
    print("✓ Realistic transaction costs: $5 per side")
    print("✓ Market slippage: 0-2 ticks")
    print("✓ Fill probability: 95%")
    print("=" * 60)
    
    # Set environment variable to enable realistic mode
    os.environ['REALISTIC_TRADING'] = '1'
    os.environ['MIN_HOLDING_PERIODS'] = '10'
    os.environ['MAX_TRADES_PER_EPISODE'] = '5'
    os.environ['EXECUTION_COST'] = '5.0'
    
    print("\n📊 Starting server with realistic constraints...")
    print("Access the dashboard at: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")
    
    # Run the standard run_local.py with our environment settings
    try:
        subprocess.run([sys.executable, "run_local.py"])
    except KeyboardInterrupt:
        print("\n\n✓ Realistic trading system stopped")

if __name__ == "__main__":
    main()