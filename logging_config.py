"""
Professional Logging Configuration for RL Trading System
Separates logs by category for easy analysis
"""
import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging():
    """Setup structured logging with separate files for each category"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for this session
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = log_dir / session_timestamp
    session_dir.mkdir(exist_ok=True)
    
    # Define log formats
    detailed_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_format = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Configure root logger to WARNING to reduce noise
    logging.getLogger().setLevel(logging.WARNING)
    
    # 1. Trading Logger - All trade executions
    trading_logger = logging.getLogger('trading')
    trading_logger.setLevel(logging.INFO)
    trading_logger.handlers.clear()
    trading_handler = logging.FileHandler(session_dir / 'trading.log')
    trading_handler.setFormatter(detailed_format)
    trading_logger.addHandler(trading_handler)
    trading_logger.propagate = False
    
    # 2. Positions Logger - Position tracking
    positions_logger = logging.getLogger('positions')
    positions_logger.setLevel(logging.INFO)
    positions_logger.handlers.clear()
    positions_handler = logging.FileHandler(session_dir / 'positions.log')
    positions_handler.setFormatter(simple_format)
    positions_logger.addHandler(positions_handler)
    positions_logger.propagate = False
    
    # 3. Rewards Logger - Reward tracking and analysis
    rewards_logger = logging.getLogger('rewards')
    rewards_logger.setLevel(logging.INFO)
    rewards_logger.handlers.clear()
    rewards_handler = logging.FileHandler(session_dir / 'rewards.log')
    rewards_handler.setFormatter(simple_format)
    rewards_logger.addHandler(rewards_handler)
    rewards_logger.propagate = False
    
    # 4. Algorithm Logger - Algorithm performance and decisions
    algorithm_logger = logging.getLogger('algorithm')
    algorithm_logger.setLevel(logging.INFO)
    # Clear any existing handlers to prevent duplicates
    algorithm_logger.handlers.clear()
    algorithm_handler = logging.FileHandler(session_dir / 'algorithm.log')
    algorithm_handler.setFormatter(detailed_format)
    algorithm_logger.addHandler(algorithm_handler)
    algorithm_logger.propagate = False
    
    # 5. Performance Logger - High-level metrics
    performance_logger = logging.getLogger('performance')
    performance_logger.setLevel(logging.INFO)
    performance_logger.handlers.clear()
    performance_handler = logging.FileHandler(session_dir / 'performance.log')
    performance_handler.setFormatter(simple_format)
    performance_logger.addHandler(performance_handler)
    performance_logger.propagate = False
    
    # Create symlink to latest session
    latest_link = log_dir / 'latest'
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(session_dir.name)
    
    return {
        'session_dir': session_dir,
        'session_timestamp': session_timestamp,
        'trading': trading_logger,
        'positions': positions_logger,
        'rewards': rewards_logger,
        'algorithm': algorithm_logger,
        'performance': performance_logger
    }

# Singleton loggers
_loggers = None

def get_loggers():
    """Get or create logger instances"""
    global _loggers
    if _loggers is None:
        _loggers = setup_logging()
    return _loggers