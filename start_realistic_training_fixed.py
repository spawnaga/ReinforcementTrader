#!/usr/bin/env python3
"""
Start realistic training with constraints - Fixed version
"""

import requests
import json
import time
import sys

def start_training():
    """Start training via API with fixed parameters"""
    base_url = "http://localhost:5000"
    
    print("\n🚀 Starting Realistic Training with Constraints (Fixed)")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Server not responding properly")
            return
        print("✅ Server is running")
    except:
        print("❌ Cannot connect to server at localhost:5000")
        return
    
    # Configuration for realistic training - without total_episodes in algo params
    config = {
        "name": "Realistic NQ Training - Fixed",
        "algorithm": "ANE_PPO",
        "parameters": {
            "total_episodes": 1000,  # This will be used by training engine, not ANE-PPO
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_range": 0.2,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "transformer_layers": 3,
            "attention_dim": 256,
            "dataConfig": {
                "symbol": "NQ",
                "startDate": "2023-01-01",
                "endDate": "2024-12-31",
                "indicators": ["RSI", "MACD", "BB", "ATR", "Volume"],
                "timeframe": "1min"
            }
        }
    }
    
    print("\n📊 Configuration:")
    print(f"   Algorithm: {config['algorithm']}")
    print(f"   Episodes: {config['parameters']['total_episodes']}")
    print(f"   Symbol: NQ Futures")
    print(f"   Constraints: Realistic (min holding, trade limits, slippage)")
    
    print("\n🔧 Starting training session...")
    
    # Start training
    try:
        response = requests.post(
            f"{base_url}/api/start_training",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get('session_id')
            print(f"\n✅ Training started successfully!")
            print(f"   Session ID: {session_id}")
            print(f"   Status: {result.get('status')}")
            
            print("\n📈 Monitor progress:")
            print(f"   From your Windows machine: python remote_monitor.py 192.168.0.129 5000")
            print(f"   Or check session status: curl http://192.168.0.129:5000/api/sessions/{session_id}")
            
            print("\n⏱️  Training is running with realistic constraints:")
            print("   - Minimum holding period: 10 time steps")
            print("   - Maximum trades per episode: 5")
            print("   - Transaction costs: $5-10 per trade")
            print("   - Slippage: 0-2 ticks")
            print("   - Fill probability: 95%")
            
            print("\n💡 Expected behavior:")
            print("   - Fewer trades (50-500 total vs 50,000+)")
            print("   - Longer holding periods")
            print("   - Realistic profit/loss patterns")
            print("   - Suitable for real market deployment")
            
            # Show initial status
            time.sleep(2)
            status_response = requests.get(f"{base_url}/api/sessions/{session_id}")
            if status_response.status_code == 200:
                session_data = status_response.json()
                print(f"\n📊 Initial Status:")
                print(f"   Current Episode: {session_data.get('current_episode')}/{session_data.get('total_episodes')}")
                print(f"   Status: {session_data.get('status')}")
                print(f"   Total Trades: {session_data.get('total_trades')}")
            
        else:
            print(f"\n❌ Failed to start training")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error starting training: {str(e)}")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    start_training()