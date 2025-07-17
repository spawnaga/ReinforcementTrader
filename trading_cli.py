#!/usr/bin/env python3
"""
Trading System CLI Interface
Comprehensive command-line interface for controlling the trading system
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
import logging

from trading_config import TradingConfig, get_config
from futures_contracts import list_available_contracts, FUTURES_SPECS
from technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description='AI Trading System - Comprehensive Trading Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train NQ futures with GPU
  python trading_cli.py --ticker NQ --device gpu --episodes 1000
  
  # Train ES futures with specific date range
  python trading_cli.py --ticker ES --start-date 2024-01-01 --end-date 2024-06-30
  
  # Use custom indicators and algorithm
  python trading_cli.py --ticker CL --indicators SMA EMA RSI MACD --algorithm ane_ppo
  
  # Specify data source
  python trading_cli.py --ticker GC --data-source ./GC/data.csv
        """
    )
    
    # Hardware options
    hardware_group = parser.add_argument_group('Hardware Options')
    hardware_group.add_argument('--device', choices=['cpu', 'gpu', 'auto'], default='auto',
                              help='Device to use for training (default: auto-detect)')
    hardware_group.add_argument('--gpu-ids', nargs='+', type=int,
                              help='Specific GPU IDs to use (e.g., --gpu-ids 0 1 2)')
    hardware_group.add_argument('--num-workers', type=int, default=4,
                              help='Number of worker threads (default: 4)')
    
    # Trading instrument options
    instrument_group = parser.add_argument_group('Trading Instrument')
    instrument_group.add_argument('--ticker', type=str, default='NQ',
                                help='Futures contract ticker (default: NQ)')
    instrument_group.add_argument('--list-contracts', action='store_true',
                                help='List all available futures contracts')
    
    # Data options
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data-source', type=str,
                          help='Data source path (default: ./{ticker}/*.csv)')
    data_group.add_argument('--start-date', type=str,
                          help='Start date for training (YYYY-MM-DD)')
    data_group.add_argument('--end-date', type=str,
                          help='End date for training (YYYY-MM-DD)')
    data_group.add_argument('--train-split', type=float, default=0.8,
                          help='Train/test split ratio (default: 0.8)')
    data_group.add_argument('--sequence-length', type=int, default=60,
                          help='Sequence length for time series (default: 60)')
    
    # Indicator options
    indicator_group = parser.add_argument_group('Technical Indicators')
    indicator_group.add_argument('--indicators', nargs='+', 
                               default=['sin_time', 'cos_time', 'sin_weekday', 'cos_weekday',
                                       'SMA', 'EMA', 'RSI', 'MACD', 'BB', 'ATR'],
                               help='Indicators to use (default: time-based + common indicators)')
    indicator_group.add_argument('--list-indicators', action='store_true',
                               help='List all available indicators with descriptions')
    indicator_group.add_argument('--indicator-params', type=str,
                               help='JSON string with indicator parameters')
    
    # Algorithm options
    algorithm_group = parser.add_argument_group('Algorithm Options')
    algorithm_group.add_argument('--algorithm', choices=['ane_ppo', 'dqn', 'q_learning', 'genetic'],
                               default='ane_ppo',
                               help='Trading algorithm to use (default: ane_ppo)')
    algorithm_group.add_argument('--use-transformer', action='store_true', default=True,
                               help='Use transformer attention (default: True)')
    algorithm_group.add_argument('--use-genetic', action='store_true', default=True,
                               help='Use genetic optimizer (default: True)')
    
    # Training options
    training_group = parser.add_argument_group('Training Options')
    training_group.add_argument('--episodes', type=int, default=1000,
                              help='Number of training episodes (default: 1000)')
    training_group.add_argument('--max-steps', type=int, default=200,
                              help='Max steps per episode (default: 200)')
    training_group.add_argument('--batch-size', type=int, default=32,
                              help='Batch size for training (default: 32)')
    training_group.add_argument('--learning-rate', type=float, default=3e-4,
                              help='Learning rate (default: 3e-4)')
    
    # Environment options
    env_group = parser.add_argument_group('Environment Options')
    env_group.add_argument('--max-trades', type=int, default=5,
                         help='Max trades per episode (default: 5)')
    env_group.add_argument('--min-holding', type=int, default=10,
                         help='Min holding periods (default: 10)')
    env_group.add_argument('--slippage', type=int, default=2,
                         help='Slippage in ticks (default: 2)')
    
    # Risk management options
    risk_group = parser.add_argument_group('Risk Management')
    risk_group.add_argument('--stop-loss', type=int, default=20,
                          help='Stop loss in ticks (default: 20)')
    risk_group.add_argument('--take-profit', type=int, default=40,
                          help='Take profit in ticks (default: 40)')
    risk_group.add_argument('--max-daily-loss', type=float, default=1000,
                          help='Max daily loss in dollars (default: 1000)')
    
    # Action options
    action_group = parser.add_argument_group('Actions')
    action_group.add_argument('--train', action='store_true',
                            help='Start training')
    action_group.add_argument('--test', action='store_true',
                            help='Run testing/evaluation')
    action_group.add_argument('--live', action='store_true',
                            help='Run live trading (requires IB connection)')
    action_group.add_argument('--config-file', type=str,
                            help='Load configuration from file')
    action_group.add_argument('--save-config', type=str,
                            help='Save current configuration to file')
    
    return parser


def list_contracts():
    """List all available futures contracts"""
    print("\nAvailable Futures Contracts:")
    print("-" * 60)
    print(f"{'Symbol':<10} {'Name':<30} {'Tick Size':<10} {'Value/Tick':<10}")
    print("-" * 60)
    
    for symbol, specs in FUTURES_SPECS.items():
        print(f"{symbol:<10} {specs['name']:<30} {specs['tick_size']:<10.5f} ${specs['value_per_tick']:<9.2f}")
    print("-" * 60)


def list_indicators():
    """List all available indicators"""
    print("\nAvailable Technical Indicators:")
    print("-" * 80)
    print(f"{'Abbreviation':<15} {'Description':<65}")
    print("-" * 80)
    
    indicators = TechnicalIndicators.list_all_indicators()
    
    # Group indicators by category
    time_based = ['sin_time', 'cos_time', 'sin_weekday', 'cos_weekday', 'hour', 'minute', 'day_of_week']
    moving_averages = ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA']
    momentum = ['RSI', 'MACD', 'STOCH', 'CCI', 'MOM', 'ROC', 'WILLR', 'MFI']
    volatility = ['ATR', 'NATR', 'BB', 'KC', 'DC']
    volume = ['OBV', 'AD', 'ADOSC', 'CMF']
    trend = ['ADX', 'AROON', 'PSAR', 'TRIX']
    
    categories = [
        ("Time-Based Indicators", time_based),
        ("Moving Averages", moving_averages),
        ("Momentum Indicators", momentum),
        ("Volatility Indicators", volatility),
        ("Volume Indicators", volume),
        ("Trend Indicators", trend)
    ]
    
    for category_name, category_indicators in categories:
        print(f"\n{category_name}:")
        for indicator in category_indicators:
            if indicator in indicators:
                print(f"  {indicator:<15} {indicators[indicator]:<65}")
    
    print("\n" + "-" * 80)


def setup_directories(config: TradingConfig, ticker: str):
    """Create directory structure for the ticker"""
    dirs = config.get_directories(ticker)
    
    logger.info(f"Setting up directories for {ticker}:")
    for name, path in dirs.items():
        logger.info(f"  {name}: {path}")
    
    return dirs


def update_config_from_args(config: TradingConfig, args):
    """Update configuration from command line arguments"""
    # Hardware settings
    config.set('hardware.device', args.device)
    if args.gpu_ids:
        config.set('hardware.gpu_ids', args.gpu_ids)
    config.set('hardware.num_workers', args.num_workers)
    
    # Instrument settings
    config.set('instrument.ticker', args.ticker)
    if args.data_source:
        config.set('instrument.data_source', args.data_source)
    
    # Data settings
    if args.start_date:
        config.set('data.start_date', args.start_date)
    if args.end_date:
        config.set('data.end_date', args.end_date)
    config.set('data.train_test_split', args.train_split)
    config.set('data.sequence_length', args.sequence_length)
    config.set('data.batch_size', args.batch_size)
    
    # Indicators
    config.set('indicators.technical', args.indicators)
    if args.indicator_params:
        params = json.loads(args.indicator_params)
        config.set('indicators.indicator_params', params)
    
    # Algorithm settings
    config.set('algorithms.primary', args.algorithm)
    config.set('algorithms.use_transformer', args.use_transformer)
    config.set('algorithms.use_genetic_optimizer', args.use_genetic)
    config.set('algorithms.algorithm_params.ane_ppo.learning_rate', args.learning_rate)
    
    # Training settings
    config.set('training.episodes', args.episodes)
    config.set('training.max_steps_per_episode', args.max_steps)
    
    # Environment settings
    config.set('environment.max_trades_per_episode', args.max_trades)
    config.set('environment.min_holding_periods', args.min_holding)
    config.set('environment.slippage_ticks', args.slippage)
    
    # Risk settings
    config.set('risk.stop_loss_ticks', args.stop_loss)
    config.set('risk.take_profit_ticks', args.take_profit)
    config.set('risk.max_daily_loss', args.max_daily_loss)


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Handle list commands
    if args.list_contracts:
        list_contracts()
        return
    
    if args.list_indicators:
        list_indicators()
        return
    
    # Load or create configuration
    if args.config_file:
        config = TradingConfig(args.config_file)
    else:
        config = get_config()
    
    # Update configuration from arguments
    update_config_from_args(config, args)
    
    # Save configuration if requested
    if args.save_config:
        config.save_config()
        logger.info(f"Configuration saved to {args.save_config}")
        return
    
    # Setup directories for the ticker
    dirs = setup_directories(config, args.ticker)
    
    # Display current configuration
    print("\nCurrent Configuration:")
    print("-" * 60)
    print(f"Ticker: {config.get('instrument.ticker')}")
    print(f"Device: {config.get('hardware.device')}")
    print(f"Algorithm: {config.get('algorithms.primary')}")
    print(f"Episodes: {config.get('training.episodes')}")
    print(f"Indicators: {', '.join(config.get_indicators())}")
    print(f"Data Directory: {dirs['data_dir']}")
    print(f"Models Directory: {dirs['models_dir']}")
    print(f"Logs Directory: {dirs['logs_dir']}")
    print("-" * 60)
    
    # Handle actions
    if args.train:
        logger.info("Starting training...")
        # Import and run training
        from run_training import run_training
        run_training(config)
    elif args.test:
        logger.info("Starting testing...")
        # Import and run testing
        from run_testing import run_testing
        run_testing(config)
    elif args.live:
        logger.info("Starting live trading...")
        # Import and run live trading
        from run_live import run_live_trading
        run_live_trading(config)
    else:
        print("\nNo action specified. Use --train, --test, or --live")
        print("Use --help for more options")


if __name__ == "__main__":
    main()