"""
Training Runner Module
Handles the training process with configuration support
"""

import logging
import sys
from pathlib import Path
import torch

from trading_config import TradingConfig
from trading_engine import TradingEngine
from futures_contracts import get_contract
from data_manager import DataManager
from technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


def setup_device(config: TradingConfig):
    """Setup compute device based on configuration"""
    device_config = config.get('hardware.device')
    
    if device_config == 'cpu':
        return 'cpu'
    elif device_config == 'gpu':
        if torch.cuda.is_available():
            gpu_ids = config.get('hardware.gpu_ids')
            if gpu_ids:
                torch.cuda.set_device(gpu_ids[0])
            return 'cuda'
        else:
            logger.warning("GPU requested but not available, falling back to CPU")
            return 'cpu'
    else:  # auto
        return 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_data(config: TradingConfig):
    """Prepare data based on configuration"""
    ticker = config.get('instrument.ticker')
    data_source = config.get('instrument.data_source')
    
    # Get contract specifications
    contract = get_contract(ticker)
    
    # Initialize data manager
    data_manager = DataManager()
    
    # Load data
    if data_source:
        logger.info(f"Loading data from {data_source}")
        df = data_manager.load_data_from_file(data_source)
    else:
        # Default path
        data_path = Path(f"./{ticker}")
        logger.info(f"Loading data from {data_path}")
        df = data_manager.load_futures_data(ticker)
    
    # Apply date filters if specified
    start_date = config.get('data.start_date')
    end_date = config.get('data.end_date')
    
    if start_date or end_date:
        df = data_manager.filter_by_date(df, start_date, end_date)
    
    # Calculate indicators
    indicators = config.get_indicators()
    indicator_params = config.get('indicators.indicator_params', {})
    
    ti = TechnicalIndicators(df)
    df_with_indicators = ti.calculate_indicators(indicators, **indicator_params)
    
    # Create train/test split
    train_split = config.get('data.train_test_split', 0.8)
    split_idx = int(len(df_with_indicators) * train_split)
    
    train_df = df_with_indicators[:split_idx]
    test_df = df_with_indicators[split_idx:]
    
    logger.info(f"Data prepared: {len(train_df)} training samples, {len(test_df)} test samples")
    logger.info(f"Features: {list(df_with_indicators.columns)}")
    
    return train_df, test_df, contract


def run_training(config: TradingConfig):
    """Run the training process"""
    logger.info("Starting training process...")
    
    # Setup device
    device = setup_device(config)
    logger.info(f"Using device: {device}")
    
    # Prepare data
    train_df, test_df, contract = prepare_data(config)
    
    # Initialize trading engine
    from app import trading_engine
    
    # Create algorithm config
    algorithm = config.get('algorithms.primary')
    algorithm_params = config.get(f'algorithms.algorithm_params.{algorithm}', {})
    
    # Add contract specifications to params
    algorithm_params.update({
        'tick_size': contract.tick_size,
        'value_per_tick': contract.value_per_tick,
        'min_holding_periods': contract.min_holding_periods,
        'execution_cost_per_order': contract.execution_cost_per_order,
        'slippage_ticks': contract.slippage_ticks
    })
    
    # Create algorithm configuration
    algo_config = {
        'algorithm': algorithm,
        'episodes': config.get('training.episodes'),
        'max_steps': config.get('training.max_steps_per_episode'),
        'batch_size': config.get('data.batch_size'),
        **algorithm_params
    }
    
    # Start training
    logger.info(f"Starting {algorithm} training for {contract.symbol}")
    logger.info(f"Algorithm config: {algo_config}")
    
    try:
        # Create training session
        session_id = trading_engine.create_training_session(
            algorithm_type=algorithm,
            config=algo_config
        )
        
        # Start training with data
        trading_engine.start_training_with_data(
            session_id=session_id,
            training_data=train_df,
            test_data=test_df,
            ticker=contract.symbol
        )
        
        logger.info(f"Training started with session ID: {session_id}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    # For testing
    from trading_config import get_config
    config = get_config()
    run_training(config)