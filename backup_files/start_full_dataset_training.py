#!/usr/bin/env python3
"""
Start training with full dataset (no shuffle, sequential processing)
"""
import requests
import json

def start_full_dataset_training():
    """Start training using the entire dataset sequentially"""
    
    print("🚀 Starting Full Dataset Training...")
    print("This will use ALL 5.3 million records sequentially")
    
    # Configuration for full dataset training
    config = {
        "session_name": "Full Dataset Sequential",
        "algorithm_type": "ANE_PPO",
        "total_episodes": 10000,        # Many episodes to go through all data
        "learning_rate": 0.0001,        # Lower learning rate for stability
        "clip_range": 0.2,              
        "entropy_coef": 0.01,           # Lower entropy for less randomness
        "n_steps": 2048,                # Larger steps
        "batch_size": 128,              # Bigger batches
        "use_full_dataset": True,       # Flag to use all data
        "shuffle_data": False,          # No shuffling - sequential
        "max_data_rows": -1             # -1 means use all available data
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
        print(f"   Dataset: FULL (5.3M records)")
        print(f"   Processing: SEQUENTIAL (no shuffle)")
        print(f"\n📊 Training Details:")
        print(f"   - Algorithm: ANE-PPO (Attention Network Enhanced PPO)")
        print(f"   - Uses transformer attention mechanisms")
        print(f"   - Multi-scale feature extraction")
        print(f"   - Genetic algorithm for hyperparameter optimization")
        print(f"\n⏱️  This will take a LONG time (days) with full dataset")
    else:
        print(f"❌ Failed to start training: {response.text}")

if __name__ == "__main__":
    start_full_dataset_training()