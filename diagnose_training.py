#!/usr/bin/env python3
"""
Diagnostic script to test the training system components
"""
import os
import sys
import time
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading"""
    logger.info("=" * 60)
    logger.info("Testing data loading...")
    
    try:
        from data_manager import DataManager
        dm = DataManager()
        
        # Try to load data
        logger.info("Loading NQ data...")
        start_time = time.time()
        data = dm.load_nq_data()
        load_time = time.time() - start_time
        
        if data is not None:
            logger.info(f"✓ Data loaded successfully in {load_time:.2f} seconds")
            logger.info(f"  Shape: {data.shape}")
            logger.info(f"  Columns: {list(data.columns)}")
            logger.info(f"  Memory usage: {data.memory_usage().sum() / 1024**2:.2f} MB")
            
            # Test limiting data
            logger.info("\nTesting data limiting...")
            limited_data = data.tail(50000)
            logger.info(f"✓ Data limited to {len(limited_data)} rows")
            
            return limited_data
        else:
            logger.error("✗ Failed to load data")
            return None
            
    except Exception as e:
        logger.error(f"✗ Error during data loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_state_creation(data):
    """Test state creation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing state creation...")
    
    if data is None:
        logger.error("No data available for state creation")
        return None
        
    try:
        from trading_engine import TradingEngine
        from gym_futures.envs.utils import TimeSeriesState
        
        engine = TradingEngine()
        
        # Test creating just 5 states
        logger.info("Creating 5 test states...")
        test_data = data.head(100)  # Just use first 100 rows
        
        start_time = time.time()
        states = []
        window_size = 60
        
        for i in range(window_size, min(window_size + 5, len(test_data))):
            logger.info(f"Creating state {i - window_size + 1}/5...")
            window_data = test_data.iloc[i - window_size:i].copy()
            
            # Add indicators
            window_data = engine._add_technical_indicators(window_data)
            
            # Ensure we have a 'time' column
            if 'time' not in window_data.columns:
                if window_data.index.name == 'timestamp':
                    window_data = window_data.reset_index()
                    window_data.rename(columns={'timestamp': 'time'}, inplace=True)
                else:
                    # Create synthetic timestamps
                    window_data['time'] = pd.date_range(start='2025-01-01', periods=len(window_data), freq='1min')
            
            # Create state
            try:
                state = TimeSeriesState(window_data)
                states.append(state)
                logger.info(f"  ✓ State created successfully")
            except Exception as e:
                logger.error(f"  ✗ Failed to create state: {str(e)}")
                
        create_time = time.time() - start_time
        logger.info(f"\n✓ Created {len(states)} states in {create_time:.2f} seconds")
        
        return states
        
    except Exception as e:
        logger.error(f"✗ Error during state creation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_environment(states):
    """Test environment creation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing environment creation...")
    
    if not states:
        logger.error("No states available for environment")
        return None
        
    try:
        from gym_futures.envs.futures_env import FuturesEnv
        
        logger.info("Creating FuturesEnv...")
        env = FuturesEnv(
            states=states,
            value_per_tick=5.0,
            tick_size=0.25,
            fill_probability=0.98,
            long_values=[-2, -1, 0, 1, 2],
            long_probabilities=[0.05, 0.15, 0.6, 0.15, 0.05],
            short_values=[-2, -1, 0, 1, 2],
            short_probabilities=[0.05, 0.15, 0.6, 0.15, 0.05],
            execution_cost_per_order=5.0
        )
        
        logger.info("✓ Environment created successfully")
        
        # Test reset
        logger.info("\nTesting environment reset...")
        state = env.reset(0)
        logger.info(f"✓ Reset successful, state shape: {state.shape if hasattr(state, 'shape') else 'TimeSeriesState'}")
        
        # Test step
        logger.info("\nTesting environment step...")
        next_state, reward, done, info = env.step(1)  # Buy action
        logger.info(f"✓ Step successful")
        logger.info(f"  Reward: {reward}")
        logger.info(f"  Done: {done}")
        logger.info(f"  Info: {info}")
        
        return env
        
    except Exception as e:
        logger.error(f"✗ Error during environment testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_algorithm(env):
    """Test algorithm creation"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing algorithm creation...")
    
    if env is None:
        logger.error("No environment available for algorithm")
        return None
        
    try:
        from rl_algorithms.ane_ppo import ANEPPO
        import torch
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        logger.info("Creating ANE-PPO algorithm...")
        algorithm = ANEPPO(
            env=env,
            device=device,
            learning_rate=3e-4,
            gamma=0.99,
            clip_range=0.2
        )
        
        logger.info("✓ Algorithm created successfully")
        
        # Test getting action
        logger.info("\nTesting action selection...")
        state = env.reset(0)
        action = algorithm.get_action(state)
        logger.info(f"✓ Action selected: {action}")
        
        return algorithm
        
    except Exception as e:
        logger.error(f"✗ Error during algorithm testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all diagnostic tests"""
    logger.info("Starting Trading System Diagnostics")
    logger.info("=" * 60)
    
    # Test each component
    data = test_data_loading()
    states = test_state_creation(data)
    env = test_environment(states)
    algorithm = test_algorithm(env)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("=" * 60)
    
    components = {
        "Data Loading": data is not None,
        "State Creation": states is not None and len(states) > 0,
        "Environment": env is not None,
        "Algorithm": algorithm is not None
    }
    
    for component, status in components.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        logger.info(f"{component}: {status_str}")
    
    if all(components.values()):
        logger.info("\n✓ All components working correctly!")
        logger.info("The issue might be in the threading or database operations.")
    else:
        logger.info("\n✗ Some components failed. Check the logs above for details.")

if __name__ == "__main__":
    main()