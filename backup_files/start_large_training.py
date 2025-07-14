#!/usr/bin/env python3
"""
Start training with more episodes and data
"""
import requests
import json

def start_large_training():
    """Start training with more episodes"""
    
    print("🚀 Starting Large-Scale Training Session...")
    
    # Configuration for extensive training
    config = {
        "session_name": "Large Scale Training",
        "algorithm_type": "ANE_PPO",
        "total_episodes": 2000,      # Much more episodes!
        "learning_rate": 0.0003,     # Slightly lower for stability
        "clip_range": 0.2,           # Standard clip range
        "entropy_coef": 0.03,        # Moderate exploration
        "n_steps": 1024,             # Larger batch for stability
        "batch_size": 64,            # Bigger batches
        "max_data_rows": 5000        # Use more data per episode
    }
    
    response = requests.post(
        "http://127.0.0.1:5000/api/start_training",
        headers={"Content-Type": "application/json"},
        data=json.dumps(config)
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Training started!")
        print(f"   Session ID: {result.get('session_id')}")
        print(f"   Total Episodes: {config['total_episodes']}")
        print(f"   Data rows: {config['max_data_rows']} per episode")
        print(f"\n📊 This session will:")
        print(f"   - Run 2000 episodes (much longer training)")
        print(f"   - Use 5000 data rows per episode")
        print(f"   - Create 83 states per episode (5000/60)")
        print(f"   - Cover more market scenarios")
        print(f"\n⏱️  Estimated time: 10-15 hours with 4 GPUs")
    else:
        print(f"❌ Failed to start training: {response.text}")

if __name__ == "__main__":
    start_large_training()