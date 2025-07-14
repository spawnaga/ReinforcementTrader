#!/usr/bin/env python3
"""
Start the Flask app and then begin realistic training
This ensures the web interface is available for monitoring
"""

import subprocess
import time
import os
import sys

def main():
    print("\n🚀 Starting Revolutionary AI Trading System")
    print("=" * 60)
    
    # First, start the Flask app in the background
    print("\n1️⃣ Starting Flask app on port 5000...")
    
    # Use run_remote_accessible.py to allow connections from other machines
    flask_process = subprocess.Popen(
        [sys.executable, "run_remote_accessible.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give Flask time to start
    print("   Waiting for Flask to initialize...")
    time.sleep(5)
    
    # Check if Flask started successfully
    flask_check = subprocess.run(
        ["curl", "-s", "http://localhost:5000/health"],
        capture_output=True,
        text=True
    )
    
    if flask_check.returncode == 0 and "healthy" in flask_check.stdout.lower():
        print("   ✅ Flask app is running!")
        print("   Access from this machine: http://localhost:5000")
        print("   Access from network: http://192.168.0.129:5000")
    else:
        print("   ❌ Flask failed to start")
        print("   Error:", flask_check.stderr)
        return
    
    print("\n2️⃣ Starting realistic training in 5 seconds...")
    print("   This will apply constraints to prevent instant trading")
    time.sleep(5)
    
    # Now start the training with constraints
    training_process = subprocess.Popen(
        [sys.executable, "start_training_with_constraints.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print("\n✅ System is running!")
    print("\n📊 Monitor from your Windows machine:")
    print("   python remote_monitor.py 192.168.0.129 5000")
    print("\n   Or use SSH tunnel:")
    print("   ssh -L 5000:localhost:5000 alex@192.168.0.129")
    print("   Then access: http://localhost:5000")
    
    print("\n⏱️  Press Ctrl+C to stop\n")
    
    try:
        # Stream training output
        for line in training_process.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        training_process.terminate()
        flask_process.terminate()
        print("✓ Stopped")

if __name__ == "__main__":
    main()