#!/usr/bin/env python3
"""
Test to reproduce the reward bug that happens at episode 7.
When a SHORT position is closed and agent opens LONG immediately,
the reward equals the previous trade's gross profit.
"""

import numpy as np
from gym_futures.envs.futures_env import FuturesEnv
from gym_futures.envs.utils import TimeSeriesState
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Create minimal test data
test_data = []
base_price = 3354.50
for i in range(100):
    # Create price movements similar to the bug scenario
    if i < 10:
        price = base_price - i * 10  # Price drops (good for SHORT)
    else:
        price = base_price - 50 + (i - 10) * 2  # Price recovers
    
    test_data.append({
        'timestamp': f'2008-01-{i+1:02d} 12:00:00',
        'open': price,
        'high': price + 5,
        'low': price - 5,
        'close': price,
        'volume': 1000
    })

# Create states
states = []
for i, row in enumerate(test_data):
    state = TimeSeriesState(
        data=[[row['timestamp'], row['open'], row['high'], row['low'], row['close'], row['volume']]],
        close_price_identifier=4,
        timestamp_identifier=0
    )
    states.append(state)

# Initialize environment
env = FuturesEnv(
    states=states[:50],  # Use first 50 states
    episode_len=50,
    tick_size=0.25,
    value_per_tick=5.0,
    commission=5.0,
    slippage_ticks=0,  # No slippage for clarity
    min_holding_periods=1,  # Allow quick trades
    max_trades_per_episode=10
)

# Set episode number for debugging
env.episode_number = 7

# Reset environment
env.reset()

# Simulate the exact pattern from the logs:
# Step 1: SELL (open SHORT)
logger.info(f"Step 1: Price ${env.states[1].price:.2f} - Action: SELL (open SHORT)")
env.step(2)  # SELL

# Step 2-6: HOLD the SHORT position
for i in range(2, 7):
    logger.info(f"Step {i}: Price ${env.states[i].price:.2f} - Action: HOLD")
    env.step(1)  # HOLD

# Step 7: BUY (close SHORT) - This is where the bug happens
logger.info(f"\nStep 7: Price ${env.states[7].price:.2f} - Action: BUY (close SHORT)")
logger.info(f"Position before BUY: {env.current_position}")
logger.info(f"Entry price: ${env.entry_price:.2f}")

# Calculate what the gross profit should be
expected_gross_profit = (env.entry_price - env.states[7].price) / 0.25 * 5.0
logger.info(f"Expected gross profit from SHORT: ${expected_gross_profit:.2f}")

# Take the BUY action
state, reward, done, info = env.step(0)  # BUY

logger.info(f"\n*** REWARD RETURNED: ${reward:.2f} ***")
logger.info(f"Position after BUY: {env.current_position}")

if abs(reward - expected_gross_profit) < 0.01:
    logger.error(f"\n!!! BUG CONFIRMED !!!")
    logger.error(f"Reward ({reward:.2f}) equals gross profit ({expected_gross_profit:.2f})")
    logger.error(f"This is the exact bug happening at episode 7!")
else:
    logger.info(f"Bug not reproduced. Reward is different from gross profit.")

# Continue for a few more steps to see aftermath
logger.info(f"\nAftermath - continuing for more steps:")
for i in range(8, 12):
    state, reward, done, info = env.step(1)  # HOLD
    logger.info(f"Step {i}: Reward ${reward:.2f}, Position: {env.current_position}")