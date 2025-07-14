#!/usr/bin/env python3
"""
Quick script to start training with realistic constraints
Run this when Flask app is already running
"""

import requests
import json
import time
import sys

def start_training():
    """Start training via API"""
    base_url = "http://localhost:5000"
    
    print("\n🚀 Starting Realistic Training with Constraints")
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
    
    # Configuration for realistic training
    config = {
        "name": "Realistic NQ Training",
        "algorithm": "ANE_PPO",
        "parameters": {
            "total_episodes": 1000,
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
            print(f"\n✅ Training started successfully!")
            print(f"   Session ID: {result.get('session_id')}")
            print(f"   Status: {result.get('status')}")
            
            print("\n📈 Monitor progress at:")
            print(f"   http://localhost:5000")
            print(f"   or from Windows: python remote_monitor.py 192.168.0.129 5000")
            
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
            
        else:
            print(f"\n❌ Failed to start training")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error starting training: {str(e)}")
        
    print("\n" + "=" * 60)

if __name__ == "__main__":
    start_training()