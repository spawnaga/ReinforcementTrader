#!/usr/bin/env python3
"""
Start training directly without API calls
"""
import os
import sys

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import the trading engine directly
from trading_engine import TradingEngine
from data_manager import DataManager

def main():
    print("\n🚀 Starting Direct Training with Realistic Constraints")
    print("=" * 60)
    
    # Initialize components
    engine = TradingEngine()
    data_manager = DataManager()
    
    # Load data
    print("\n📊 Loading NQ futures data...")
    data = data_manager.load_nq_data(start_date="2023-01-01", end_date="2024-12-31")
    if data is None:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(data):,} records")
    
    # Start training with realistic environment
    config = {
        "name": "Direct Realistic Training",
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
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
            "genetic_population_size": 50,
            "genetic_mutation_rate": 0.1,
            "genetic_crossover_rate": 0.8,
            "attention_heads": 8,
            "attention_dim": 256,
            "transformer_layers": 3
        }
    }
    
    print("\n🔧 Using RealisticFuturesEnv with constraints:")
    print("   - Max 5 trades per episode")
    print("   - Min 10 time steps holding period")
    print("   - $5 execution cost per side")
    print("   - 0-2 tick slippage")
    
    # Start training
    session_id = engine.start_training(
        algorithm="ANE_PPO",
        parameters=config["parameters"],
        data_config={
            "symbol": "NQ",
            "startDate": "2023-01-01", 
            "endDate": "2024-12-31",
            "indicators": ["RSI", "MACD", "BB", "ATR", "Volume"],
            "timeframe": "1min"
        }
    )
    
    if session_id:
        print(f"\n✅ Training started! Session ID: {session_id}")
        print("\nMonitor progress with:")
        print(f"   python monitor_training.py {session_id}")
    else:
        print("\n❌ Failed to start training")

if __name__ == "__main__":
    main()