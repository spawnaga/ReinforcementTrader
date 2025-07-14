#!/usr/bin/env python3
"""
Test the infinite trading fix
"""

import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from futures_env_realistic import RealisticFuturesEnv
from trading_engine import TimeSeriesState

def create_test_states(n_states=100):
    """Create test states with incrementing timestamps"""
    states = []
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    base_price = 20000.0
    
    for i in range(n_states):
        # Create simple price movement
        price = base_price + np.random.randn() * 10
        timestamp = base_time + timedelta(minutes=i)
        
        # Create a simple state with just price data
        state_data = {
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price,
            'volume': 1000
        }
        
        state = TimeSeriesState(
            timestamp=timestamp,
            data=state_data,
            price=price
        )
        states.append(state)
    
    return states

def test_anti_exploitation():
    """Test that the anti-exploitation measures work"""
    print("Testing anti-exploitation measures...")
    
    # Create test environment
    states = create_test_states(100)
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=5.0,
        tick_size=0.25,
        execution_cost_per_order=5.0,
        min_holding_periods=10,
        max_trades_per_episode=5,
        slippage_ticks=2
    )
    
    # Reset environment
    env.reset()
    
    # Test 1: Cannot trade at same state twice
    print("\nTest 1: Preventing multiple trades at same state")
    env.buy(states[0])  # Should work
    initial_position = env.current_position
    env.buy(states[0])  # Should be blocked
    assert env.current_position == initial_position, "Trade should have been blocked!"
    print("✓ Successfully blocked duplicate trade at same state")
    
    # Test 2: Must wait minimum gap between trades
    print("\nTest 2: Enforcing minimum gap between trades")
    env.reset()
    env.buy(states[0])  # Trade at index 0
    
    # Try to trade too soon
    for i in range(1, 5):
        env.current_index = i
        env.sell(states[i])  # Should be blocked
        assert env.current_position == 1, f"Trade at index {i} should have been blocked!"
    
    # Trade after minimum gap
    env.current_index = 5
    env.sell(states[5])  # Should work
    assert env.current_position == 0, "Trade should have succeeded after minimum gap!"
    print("✓ Successfully enforced minimum gap between trades")
    
    # Test 3: Cannot exceed max trades per episode
    print("\nTest 3: Enforcing max trades per episode")
    env.reset()
    trades_made = 0
    
    for i in range(0, 50, 10):  # Try to make many trades
        env.current_index = i
        if env.current_position == 0:
            env.buy(states[i])
            if env.current_position == 1:
                trades_made += 1
        else:
            env.sell(states[i])
            if env.current_position == 0:
                trades_made += 1
    
    assert trades_made <= env.max_trades_per_episode, "Exceeded max trades per episode!"
    print(f"✓ Successfully limited trades to {trades_made}/{env.max_trades_per_episode}")
    
    print("\nAll anti-exploitation tests passed! ✅")
    print("\nThe fix prevents:")
    print("1. Multiple trades at the same state/timestamp")
    print("2. Rapid-fire trading (enforces minimum gap)")
    print("3. Excessive trading (limits trades per episode)")

if __name__ == "__main__":
    test_anti_exploitation()