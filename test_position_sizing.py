
#!/usr/bin/env python3
"""
Test script to verify position sizing functionality
"""

import numpy as np
from gym_futures.envs.utils import TimeSeriesState
from futures_env_realistic import RealisticFuturesEnv

def create_test_states():
    """Create simple test states for testing"""
    states = []
    for i in range(100):
        # Create a simple data array: [timestamp, price, volume]
        timestamp = f"2023-01-01 00:{(i // 60):02d}:{(i % 60):02d}"
        data = np.array([[timestamp, 100.0 + i * 0.1, 1000]])
        state = TimeSeriesState(data, timestamp_identifier=0, close_price_identifier=1)
        states.append(state)
    return states

def test_position_sizing():
    """Test that position sizing works correctly"""
    print("Testing position sizing functionality...")

    # Create test environment
    states = create_test_states()
    env = RealisticFuturesEnv(
        states=states,
        value_per_tick=50,
        tick_size=0.25,
        fill_probability=1.0,
        execution_cost_per_order=5.0,
        min_holding_periods=5,
        max_trades_per_episode=3,
        slippage_ticks=1,

        default_position_size=2,  # Start with 2 contracts
        enable_trading_logger=False

    )

    # Test reset
    state = env.reset()
    print(f"Initial position size: {env.current_position_size}")

    # Test that position size is updated
    env._win_streak = 3  # Simulate 3 consecutive wins
    env._update_position_size()
    print(f"Position size after 3 wins: {env.current_position_size}")

    env._win_streak = 0  # Simulate losses
    env._update_position_size()
    print(f"Position size after losses: {env.current_position_size}")

    # Test with different win streak values
    env._win_streak = 5
    env._update_position_size()
    print(f"Position size after 5 wins: {env.current_position_size}")

    env._win_streak = -1  # Simulate losses
    env._update_position_size()
    print(f"Position size after negative streak: {env.current_position_size}")

    # Test trading with position size
    print("\nTesting trading with position size...")

    # Simulate a buy
    env.buy(states[0])
    print(f"After buy: position={env.current_position}, position_size={env.current_position_size}")

    # Simulate a sell
    env.sell(states[1])
    print(f"After sell: position={env.current_position}, position_size={env.current_position_size}")

    print("Position sizing test completed successfully!")

if __name__ == "__main__":
    test_position_sizing()
