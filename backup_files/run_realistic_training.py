#!/usr/bin/env python3
"""
Run training with realistic constraints to prevent instant trading exploitation
This ensures the AI learns strategies that work in real markets
"""

import os
import sys
from datetime import datetime
from app import app
from extensions import db
from models import TradingSession
from trading_engine import TradingEngine
from data_manager import DataManager

def setup_realistic_environment():
    """Configure the training environment with realistic constraints"""
    print("\n🚀 Starting Realistic Trading Training")
    print("=" * 60)
    print("This training includes:")
    print("✓ Minimum holding periods (10 time steps)")
    print("✓ Realistic transaction costs ($5 per side)")
    print("✓ Market slippage simulation (0-2 ticks)")
    print("✓ Trade frequency limits (max 5 per episode)")
    print("✓ 95% fill probability")
    print("=" * 60)
    
    # Create patched environment settings
    realistic_settings = {
        'min_holding_periods': 10,  # Must hold for 10 time steps minimum
        'max_trades_per_episode': 5,  # Maximum 5 trades per episode
        'execution_cost_per_order': 5.0,  # $5 per side (realistic commission)
        'slippage_ticks': 2,  # 0-2 tick slippage
        'fill_probability': 0.95,  # 95% of orders fill
    }
    
    return realistic_settings

def patch_futures_env():
    """Monkey patch the FuturesEnv to add realistic constraints"""
    import gym_futures.envs.futures_env as futures_env
    
    # Store original __init__
    original_init = futures_env.FuturesEnv.__init__
    
    # Create patched __init__
    def patched_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        
        # Add realistic constraints
        self.min_holding_periods = kwargs.get('min_holding_periods', 10)
        self.max_trades_per_episode = kwargs.get('max_trades_per_episode', 5)
        self.slippage_ticks = kwargs.get('slippage_ticks', 2)
        self.holding_time = 0
        self.trades_this_episode = 0
        
        # Override execution cost
        self.execution_cost_per_order = kwargs.get('execution_cost_per_order', 5.0)
        
        print(f"✓ Environment patched with realistic constraints")
        print(f"  - Min holding: {self.min_holding_periods} steps")
        print(f"  - Max trades: {self.max_trades_per_episode} per episode")
        print(f"  - Execution cost: ${self.execution_cost_per_order} per side")
    
    # Store original buy/sell methods
    original_buy = futures_env.FuturesEnv.buy
    original_sell = futures_env.FuturesEnv.sell
    original_step = futures_env.FuturesEnv.step
    original_reset = futures_env.FuturesEnv.reset
    
    # Create patched buy method
    def patched_buy(self, state):
        # Check if we can trade
        if hasattr(self, 'trades_this_episode') and self.trades_this_episode >= self.max_trades_per_episode:
            return  # Trade limit reached
        
        if self.current_position != 0 and hasattr(self, 'holding_time') and self.holding_time < self.min_holding_periods:
            return  # Haven't held long enough
        
        # Apply slippage if we have the attribute
        if hasattr(self, 'slippage_ticks'):
            import numpy as np
            original_generate = self.generate_random_fill_differential
            
            def slippage_generate(price, direction):
                slippage = np.random.randint(0, self.slippage_ticks + 1) * self.tick_size
                if direction == 1:  # Buying
                    return price + slippage
                else:
                    return price - slippage
            
            self.generate_random_fill_differential = slippage_generate
        
        # Track position changes
        old_position = self.current_position
        original_buy(self, state)
        
        # If position changed, reset holding time and increment trade count
        if self.current_position != old_position:
            if hasattr(self, 'holding_time'):
                self.holding_time = 0
            if hasattr(self, 'trades_this_episode') and old_position != 0:
                self.trades_this_episode += 1
    
    # Create patched sell method
    def patched_sell(self, state):
        # Check if we can trade
        if hasattr(self, 'trades_this_episode') and self.trades_this_episode >= self.max_trades_per_episode:
            return  # Trade limit reached
        
        if self.current_position != 0 and hasattr(self, 'holding_time') and self.holding_time < self.min_holding_periods:
            return  # Haven't held long enough
        
        # Apply slippage
        if hasattr(self, 'slippage_ticks'):
            import numpy as np
            original_generate = self.generate_random_fill_differential
            
            def slippage_generate(price, direction):
                slippage = np.random.randint(0, self.slippage_ticks + 1) * self.tick_size
                if direction == 1:
                    return price + slippage
                else:
                    return price - slippage
            
            self.generate_random_fill_differential = slippage_generate
        
        # Track position changes
        old_position = self.current_position
        original_sell(self, state)
        
        # If position changed, reset holding time and increment trade count
        if self.current_position != old_position:
            if hasattr(self, 'holding_time'):
                self.holding_time = 0
            if hasattr(self, 'trades_this_episode') and old_position != 0:
                self.trades_this_episode += 1
    
    # Create patched step method
    def patched_step(self, action):
        # Update holding time
        if hasattr(self, 'holding_time') and self.current_position != 0:
            self.holding_time += 1
        
        return original_step(self, action)
    
    # Create patched reset method
    def patched_reset(self, *args, **kwargs):
        result = original_reset(self, *args, **kwargs)
        
        # Reset episode tracking
        if hasattr(self, 'holding_time'):
            self.holding_time = 0
        if hasattr(self, 'trades_this_episode'):
            self.trades_this_episode = 0
        
        return result
    
    # Apply patches
    futures_env.FuturesEnv.__init__ = patched_init
    futures_env.FuturesEnv.buy = patched_buy
    futures_env.FuturesEnv.sell = patched_sell
    futures_env.FuturesEnv.step = patched_step
    futures_env.FuturesEnv.reset = patched_reset
    
    print("\n✓ FuturesEnv successfully patched with realistic constraints")

def main():
    """Run training with realistic constraints"""
    with app.app_context():
        # Apply the patch before importing trading engine
        patch_futures_env()
        
        # Get realistic settings
        settings = setup_realistic_environment()
        
        # Initialize components
        engine = TradingEngine()
        data_manager = DataManager()
        
        print("\n📊 Loading NQ futures data...")
        data = data_manager.load_nq_data()
        
        if data is None or data.empty:
            print("❌ No data available for training")
            return
        
        print(f"✓ Loaded {len(data):,} records")
        
        # Check for existing active sessions
        active_session = TradingSession.query.filter_by(status='active').first()
        
        if active_session:
            print(f"\n⚠️  Active session found: {active_session.session_name}")
            print("Options:")
            print("1. Stop it and start realistic training")
            print("2. Exit without changes")
            
            choice = input("\nEnter choice (1 or 2): ").strip()
            
            if choice == '1':
                # Stop the existing session
                active_session.status = 'completed'
                db.session.commit()
                print("✓ Previous session stopped")
            else:
                print("Exiting without changes")
                return
        
        # Create new realistic training session
        print("\n🎯 Starting realistic training session...")
        
        # Create a new training session in the database first
        session_name = f"Realistic Training {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        try:
            # Create session in database
            new_session = TradingSession(
                session_name=session_name,
                algorithm_type='ANE-PPO',
                status='pending',
                start_time=datetime.now(),
                total_episodes=1000,  # Start with fewer episodes to test
                current_episode=0,
                total_profit=0.0,
                total_trades=0,
                win_rate=0.0
            )
            db.session.add(new_session)
            db.session.commit()
            
            session_id = new_session.id
            print(f"\n✓ Created training session ID: {session_id}")
            
            # Prepare configuration for training
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
                    'max_grad_norm': 0.5,
                    # These will be used by our patched environment
                    'min_holding_periods': settings['min_holding_periods'],
                    'max_trades_per_episode': settings['max_trades_per_episode'],
                    'execution_cost_per_order': settings['execution_cost_per_order'],
                    'slippage_ticks': settings['slippage_ticks'],
                    'fill_probability': settings['fill_probability']
                }
            }
            
            # Start training with the session ID and config
            success = engine.start_training(session_id, config)
            
            if success:
                print("\n✅ Realistic training started successfully!")
                print("\nExpected behavior:")
                print("- Fewer trades (max 5 per episode)")
                print("- Longer holding periods (min 10 time steps)")
                print("- More realistic profits/losses")
                print("- Better preparation for live trading")
                
                print("\nMonitor progress with:")
                print("  python monitor_training_enhanced.py")
                print("\nOr from Windows:")
                print("  python remote_monitor.py 192.168.0.129 5000")
            else:
                print("\n❌ Failed to start training - another session may be active")
                
        except Exception as e:
            print(f"\n❌ Error starting training: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Ensure we're in the right directory
    if not os.path.exists('app.py'):
        print("❌ Please run this from the ReinforcementTrader directory")
        sys.exit(1)
    
    main()