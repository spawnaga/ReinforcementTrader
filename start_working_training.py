#!/usr/bin/env python3
"""
Start training with correct ANE-PPO parameters
"""

import requests
import json
import time

def start_training():
    """Start training with correct parameter names"""
    base_url = "http://localhost:5000"
    
    print("\n🚀 Starting Realistic Training - Correct Parameters")
    print("=" * 60)
    
    # Check server
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code != 200:
            print("❌ Server not responding")
            return
        print("✅ Server is running")
    except:
        print("❌ Cannot connect to server")
        return
    
    # Use correct ANE-PPO parameter names
    config = {
        "name": "Realistic NQ Training - Working",
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
            "entropy_coef": 0.01,  # Correct parameter name
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "genetic_population_size": 50,
            "genetic_mutation_rate": 0.1,
            "genetic_crossover_rate": 0.8,
            "attention_heads": 8,
            "attention_dim": 256,
            "transformer_layers": 3,
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
    print(f"   Algorithm: ANE-PPO")
    print(f"   Episodes: 1000")
    print(f"   Learning Rate: 0.0003")
    print(f"   Transformer Layers: 3")
    print(f"   Attention Dimension: 256")
    
    print("\n🔧 Starting training...")
    
    try:
        response = requests.post(
            f"{base_url}/api/start_training",
            json=config,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            session_id = result.get('session_id')
            print(f"\n✅ Training started!")
            print(f"   Session ID: {session_id}")
            
            print("\n📈 Monitor from Windows:")
            print(f"   python remote_monitor.py 192.168.0.129 5000")
            
            print("\n⏱️  Realistic Constraints Active:")
            print("   - Min holding: 10 steps")
            print("   - Max trades/episode: 5")
            print("   - Costs: $5-10 per trade")
            print("   - Slippage: 0-2 ticks")
            
            # Check status after 3 seconds
            time.sleep(3)
            status_resp = requests.get(f"{base_url}/api/sessions/{session_id}")
            if status_resp.status_code == 200:
                data = status_resp.json()
                print(f"\n📊 Status Update:")
                print(f"   Episode: {data.get('current_episode')}/1000")
                print(f"   Status: {data.get('status')}")
                print(f"   Trades: {data.get('total_trades')}")
                
        else:
            print(f"\n❌ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    start_training()