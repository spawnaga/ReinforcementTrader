#!/usr/bin/env python3
"""
Diagnostic script to test the training system components
"""
import os
import sys
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_data_loading():
    """Test data loading"""
    logger.info("Testing data loading...")
    try:
        from data_manager import DataManager
        dm = DataManager()
        
        # Load NQ data
        data = dm.load_nq_data()
        if data is not None:
            logger.info(f"✓ Data loaded: {len(data)} rows")
            logger.info(f"  Columns: {list(data.columns)}")
            logger.info(f"  Date range: {data.index[0]} to {data.index[-1]}")
            return data
        else:
            logger.error("✗ Failed to load data")
            return None
    except Exception as e:
        logger.error(f"✗ Data loading error: {e}")
        return None

def test_state_creation(data):
    """Test state creation"""
    logger.info("\nTesting state creation...")
    try:
        from gym_futures.envs.utils import TimeSeriesState
        
        # Create states from data
        states = []
        for i in range(min(10, len(data))):
            state = TimeSeriesState(data.iloc[i:i+1])
            states.append(state)
            
        logger.info(f"✓ Created {len(states)} states")
        if states:
            logger.info(f"  State shape: {states[0].shape}")
            logger.info(f"  State features: {states[0].feature_names[:5]}...")
        return states
    except Exception as e:
        logger.error(f"✗ State creation error: {e}")
        return None

def test_environment(states):
    """Test environment creation"""
    logger.info("\nTesting environment...")
    try:
        from gym_futures.envs.futures_env import FuturesEnv
        
        env = FuturesEnv(
            states=states,
            value_per_tick=5.0,
            tick_size=0.25,
            fill_probability=0.95,
            execution_cost_per_order=5.0,
            session_id=999  # Test session
        )
        
        logger.info("✓ Environment created")
        
        # Test reset
        obs = env.reset()
        logger.info(f"  Reset observation shape: {obs.shape}")
        
        # Test step
        action = 1  # Buy
        obs, reward, done, info = env.step(action)
        logger.info(f"  Step completed: reward={reward}, done={done}")
        
        return env
    except Exception as e:
        logger.error(f"✗ Environment error: {e}")
        return None

def test_algorithm(env):
    """Test algorithm creation"""
    logger.info("\nTesting algorithm...")
    try:
        from rl_algorithms.ane_ppo import ANEPPO
        
        algo = ANEPPO(env)
        logger.info("✓ ANE-PPO algorithm created")
        logger.info(f"  Device: {algo.device}")
        logger.info(f"  Input dim: {algo.actor_critic.feature_extractors[0][0].in_features}")
        
        return algo
    except Exception as e:
        logger.error(f"✗ Algorithm error: {e}")
        return None

def check_database():
    """Check database sessions and trades"""
    logger.info("\nChecking database...")
    try:
        db_url = os.environ.get('DATABASE_URL')
        if not db_url:
            logger.error("✗ DATABASE_URL not set")
            return
            
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Check sessions
            result = conn.execute(text("SELECT id, session_name, status, current_episode, total_episodes FROM trading_session ORDER BY id DESC LIMIT 5"))
            sessions = result.fetchall()
            
            logger.info(f"\nRecent sessions:")
            for s in sessions:
                logger.info(f"  ID: {s[0]}, Name: {s[1]}, Status: {s[2]}, Episodes: {s[3]}/{s[4]}")
            
            # Check trades
            result = conn.execute(text("SELECT COUNT(*) FROM trade"))
            trade_count = result.scalar()
            logger.info(f"\nTotal trades in database: {trade_count}")
            
            # Check recent trades
            result = conn.execute(text("SELECT id, session_id, entry_price, exit_price, profit_loss FROM trade ORDER BY id DESC LIMIT 5"))
            trades = result.fetchall()
            
            if trades:
                logger.info("\nRecent trades:")
                for t in trades:
                    logger.info(f"  Trade {t[0]}: Session {t[1]}, Entry: {t[2]}, Exit: {t[3]}, P&L: {t[4]}")
                    
    except Exception as e:
        logger.error(f"✗ Database error: {e}")

def main():
    """Run all diagnostic tests"""
    logger.info("=== Trading System Diagnostics ===")
    logger.info(f"Time: {datetime.now()}")
    
    # Check database first
    check_database()
    
    # Test components
    data = test_data_loading()
    if data is not None:
        states = test_state_creation(data)
        if states:
            env = test_environment(states)
            if env:
                algo = test_algorithm(env)
    
    logger.info("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    main()