#!/usr/bin/env python3
"""
Start training with realistic constraints directly
This script creates a training session and starts it with proper constraints
Run this on your remote machine to fix the instant trading issue
"""

import os
import sys
import time
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app
from extensions import db
from models import TradingSession
from trading_engine import TradingEngine
from data_manager import DataManager

def apply_realistic_patches():
    """Apply patches to FuturesEnv to add realistic constraints"""
    import gym_futures.envs.futures_env as futures_env
    
    print("🔧 Applying realistic trading patches...")
    
    # Store original methods
    original_init = futures_env.FuturesEnv.__init__
    original_buy = futures_env.FuturesEnv.buy
    original_sell = futures_env.FuturesEnv.sell
    original_step = futures_env.FuturesEnv.step
    original_reset = futures_env.FuturesEnv.reset
    
    # Patch __init__ to add constraint attributes
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        
        # Add realistic trading constraints
        self.min_holding_periods = 10  # Must hold for 10 time steps
        self.max_trades_per_episode = 5  # Max 5 trades per episode
        self.holding_time = 0
        self.trades_this_episode = 0
        
        # Override execution cost to be more realistic
        self.execution_cost_per_order = 5.0  # $5 per side
        
        print(f"  ✓ Min holding period: {self.min_holding_periods} steps")
        print(f"  ✓ Max trades per episode: {self.max_trades_per_episode}")
        print(f"  ✓ Execution cost: ${self.execution_cost_per_order} per side")
    
    # Patch buy method
    def patched_buy(self, state):
        # Check constraints
        if self.trades_this_episode >= self.max_trades_per_episode:
            return  # Trade limit reached
        
        if self.current_position != 0 and self.holding_time < self.min_holding_periods:
            return  # Haven't held long enough
        
        # Track position changes
        old_position = self.current_position
        original_buy(self, state)
        
        # Update tracking if position changed
        if self.current_position != old_position:
            self.holding_time = 0
            if old_position != 0:  # Closing a position counts as a trade
                self.trades_this_episode += 1
    
    # Patch sell method
    def patched_sell(self, state):
        # Check constraints
        if self.trades_this_episode >= self.max_trades_per_episode:
            return  # Trade limit reached
        
        if self.current_position != 0 and self.holding_time < self.min_holding_periods:
            return  # Haven't held long enough
        
        # Track position changes
        old_position = self.current_position
        original_sell(self, state)
        
        # Update tracking if position changed
        if self.current_position != old_position:
            self.holding_time = 0
            if old_position != 0:  # Closing a position counts as a trade
                self.trades_this_episode += 1
    
    # Patch step method to track holding time
    def patched_step(self, action):
        # Update holding time if in a position
        if self.current_position != 0:
            self.holding_time += 1
        
        return original_step(self, action)
    
    # Patch reset method
    def patched_reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        
        # Reset episode tracking
        self.holding_time = 0
        self.trades_this_episode = 0
        
        return result
    
    # Apply all patches
    futures_env.FuturesEnv.__init__ = patched_init
    futures_env.FuturesEnv.buy = patched_buy
    futures_env.FuturesEnv.sell = patched_sell
    futures_env.FuturesEnv.step = patched_step
    futures_env.FuturesEnv.reset = patched_reset
    
    print("✅ Realistic trading patches applied successfully!")

def main():
    """Main function to start training with constraints"""
    
    print("\n🚀 Revolutionary AI Trading System - Realistic Mode")
    print("=" * 60)
    print("Starting training with realistic constraints:")
    print("✓ Minimum holding period: 10 time steps")
    print("✓ Maximum trades per episode: 5")
    print("✓ Realistic transaction costs: $5 per side")
    print("✓ No instant trading exploitation")
    print("=" * 60)
    
    # Apply patches before creating any environments
    apply_realistic_patches()
    
    with app.app_context():
        # Initialize components
        engine = TradingEngine()
        data_manager = DataManager()
        
        print("\n📊 Loading NQ futures data...")
        data = data_manager.load_nq_data()
        
        if data is None or data.empty:
            print("❌ No data available for training")
            return
        
        print(f"✓ Loaded {len(data):,} records")
        
        # Check for active sessions
        active_session = TradingSession.query.filter_by(status='active').first()
        if active_session:
            print(f"\n⚠️  Active session found: {active_session.session_name}")
            response = input("Stop it and start new realistic training? (y/n): ")
            
            if response.lower() == 'y':
                active_session.status = 'completed'
                db.session.commit()
                print("✓ Previous session stopped")
                time.sleep(2)  # Give time for threads to clean up
            else:
                print("Exiting without changes")
                return
        
        # Create new training session
        session_name = f"Realistic Training {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        print(f"\n🎯 Creating session: {session_name}")
        
        new_session = TradingSession(
            session_name=session_name,
            algorithm_type='ANE-PPO',
            status='pending',
            start_time=datetime.now(),
            total_episodes=1000,
            current_episode=0,
            total_profit=0.0,
            total_trades=0,
            win_rate=0.0
        )
        db.session.add(new_session)
        db.session.commit()
        
        session_id = new_session.id
        print(f"✓ Created session ID: {session_id}")
        
        # Training configuration
        config = {
            'algorithm_type': 'ANE-PPO',
            'symbol': 'NQ',
            'total_episodes': 1000,
            'parameters': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'clip_range': 0.2,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'entropy_coef': 0.01,  # Fixed parameter name
                'value_loss_coef': 0.5,  # Fixed parameter name
                'max_grad_norm': 0.5
            }
        }
        
        print("\n🏃 Starting training...")
        success = engine.start_training(session_id, config)
        
        if success:
            print("\n✅ Training started successfully!")
            print("\n📊 Expected results with realistic constraints:")
            print("  • 50-500 trades total (not 50,000+)")
            print("  • Realistic profit/loss patterns")
            print("  • Strategies that work in real markets")
            print("  • No instant trading exploitation")
            
            print("\n📈 Monitor progress:")
            print(f"  • Local: http://localhost:5000")
            print(f"  • Remote: http://192.168.0.129:5000")
            print("\n  Or use: python monitor_training_enhanced.py")
            
            print("\n⏱️  Training will run in background")
            print("Press Ctrl+C to stop the server")
            
            # Keep the script running
            try:
                while True:
                    time.sleep(10)
                    # Could add periodic status updates here
            except KeyboardInterrupt:
                print("\n\n✓ Training continues in background")
                print("Access the dashboard to monitor progress")
        else:
            print("\n❌ Failed to start training")
            print("Another session may be active or there was an error")

if __name__ == "__main__":
    main()