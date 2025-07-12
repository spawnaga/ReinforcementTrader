"""
Comprehensive Trading Logger for the AI Trading System

This module provides detailed logging of all trading activities including:
- Trade entries and exits with exact prices
- Position changes
- Reward calculations
- Trading errors and warnings
- Performance metrics
"""
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path

class TradingLogger:
    """
    Comprehensive logger for all trading activities
    """
    
    def __init__(self, log_dir: str = "logs/trading"):
        """
        Initialize the trading logger
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup different loggers for different aspects
        self.setup_loggers()
        
        # Trading history storage
        self.trades = []
        self.positions = []
        self.rewards = []
        self.errors = []
        
    def setup_loggers(self):
        """Setup specialized loggers for different trading aspects"""
        
        # Main trading logger
        self.trade_logger = self._create_logger(
            'trading',
            f'trading_{self.session_timestamp}.log',
            level=logging.INFO
        )
        
        # Position logger
        self.position_logger = self._create_logger(
            'positions',
            f'positions_{self.session_timestamp}.log',
            level=logging.INFO
        )
        
        # Reward logger
        self.reward_logger = self._create_logger(
            'rewards',
            f'rewards_{self.session_timestamp}.log',
            level=logging.INFO
        )
        
        # Error logger
        self.error_logger = self._create_logger(
            'errors',
            f'errors_{self.session_timestamp}.log',
            level=logging.WARNING
        )
        
        # Debug logger
        self.debug_logger = self._create_logger(
            'debug',
            f'debug_{self.session_timestamp}.log',
            level=logging.DEBUG
        )
        
    def _create_logger(self, name: str, filename: str, level=logging.INFO) -> logging.Logger:
        """Create a logger with file handler"""
        logger = logging.getLogger(f"trading.{name}")
        logger.setLevel(level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename)
        file_handler.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        return logger
    
    def log_trade_entry(self, 
                       timestamp: datetime,
                       position_type: str,
                       entry_price: Optional[float],
                       target_price: Optional[float],
                       quantity: int = 1,
                       session_id: Optional[int] = None,
                       state_info: Optional[Dict] = None):
        """Log trade entry details"""
        
        trade_info = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'action': 'ENTRY',
            'position_type': position_type,
            'entry_price': entry_price,
            'target_price': target_price,
            'quantity': quantity,
            'session_id': session_id,
            'state_info': state_info
        }
        
        # Check for None prices
        if entry_price is None:
            self.error_logger.error(f"CRITICAL: Entry price is None for {position_type} trade at {timestamp}")
            trade_info['error'] = 'Entry price is None'
        
        self.trades.append(trade_info)
        self.trade_logger.info(f"TRADE ENTRY: {json.dumps(trade_info, indent=2)}")
        
        # Also log to debug
        self.debug_logger.debug(f"Detailed state at entry: {json.dumps(state_info or {}, indent=2)}")
        
    def log_trade_exit(self,
                      timestamp: datetime,
                      position_type: str,
                      entry_price: Optional[float],
                      exit_price: Optional[float],
                      profit_loss: Optional[float] = None,
                      session_id: Optional[int] = None,
                      state_info: Optional[Dict] = None):
        """Log trade exit details"""
        
        trade_info = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'action': 'EXIT',
            'position_type': position_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss': profit_loss,
            'session_id': session_id,
            'state_info': state_info
        }
        
        # Check for None prices
        if entry_price is None or exit_price is None:
            self.error_logger.error(
                f"CRITICAL: Missing prices - Entry: {entry_price}, Exit: {exit_price} at {timestamp}"
            )
            trade_info['error'] = 'Missing price data'
        
        self.trades.append(trade_info)
        self.trade_logger.info(f"TRADE EXIT: {json.dumps(trade_info, indent=2)}")
        
    def log_position_change(self,
                           timestamp: datetime,
                           old_position: int,
                           new_position: int,
                           reason: str,
                           price: Optional[float] = None):
        """Log position changes"""
        
        position_info = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'old_position': old_position,
            'new_position': new_position,
            'reason': reason,
            'price': price
        }
        
        self.positions.append(position_info)
        self.position_logger.info(f"POSITION CHANGE: {json.dumps(position_info, indent=2)}")
        
    def log_reward_calculation(self,
                              timestamp: datetime,
                              reward: float,
                              position: int,
                              entry_price: Optional[float],
                              exit_price: Optional[float],
                              calculation_details: Optional[Dict] = None):
        """Log reward calculations"""
        
        reward_info = {
            'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
            'reward': reward,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'calculation_details': calculation_details
        }
        
        # Check for issues
        if entry_price is None or exit_price is None:
            self.error_logger.warning(
                f"Reward calculation with None prices - Entry: {entry_price}, Exit: {exit_price}"
            )
            reward_info['warning'] = 'Calculated with None prices'
        
        self.rewards.append(reward_info)
        self.reward_logger.info(f"REWARD: {json.dumps(reward_info, indent=2)}")
        
    def log_error(self, error_type: str, error_message: str, context: Optional[Dict] = None):
        """Log trading errors"""
        
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'context': context
        }
        
        self.errors.append(error_info)
        self.error_logger.error(f"ERROR: {json.dumps(error_info, indent=2)}")
        
    def log_state_debug(self, state_type: str, state_data: Any):
        """Log detailed state information for debugging"""
        
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'state_type': state_type,
            'state_data': str(state_data)
        }
        
        self.debug_logger.debug(f"STATE DEBUG: {json.dumps(debug_info, indent=2)}")
        
    def generate_trading_report(self) -> Dict[str, Any]:
        """Generate a comprehensive trading report"""
        
        report = {
            'session_timestamp': self.session_timestamp,
            'total_trades': len([t for t in self.trades if t['action'] == 'EXIT']),
            'total_errors': len(self.errors),
            'trades_with_none_prices': len([t for t in self.trades if 'error' in t]),
            'position_changes': len(self.positions),
            'rewards_calculated': len(self.rewards),
            'summary': {
                'successful_trades': 0,
                'failed_trades': 0,
                'total_profit_loss': 0.0
            }
        }
        
        # Calculate successful trades
        for trade in self.trades:
            if trade['action'] == 'EXIT' and 'error' not in trade:
                report['summary']['successful_trades'] += 1
                if trade.get('profit_loss'):
                    report['summary']['total_profit_loss'] += trade['profit_loss']
            elif 'error' in trade:
                report['summary']['failed_trades'] += 1
        
        # Save report
        report_path = self.log_dir / f'report_{self.session_timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.trade_logger.info(f"REPORT GENERATED: {json.dumps(report, indent=2)}")
        
        return report
    
    def save_to_csv(self):
        """Save trading data to CSV files for analysis"""
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(self.log_dir / f'trades_{self.session_timestamp}.csv', index=False)
        
        # Save positions
        if self.positions:
            positions_df = pd.DataFrame(self.positions)
            positions_df.to_csv(self.log_dir / f'positions_{self.session_timestamp}.csv', index=False)
        
        # Save rewards
        if self.rewards:
            rewards_df = pd.DataFrame(self.rewards)
            rewards_df.to_csv(self.log_dir / f'rewards_{self.session_timestamp}.csv', index=False)
        
        # Save errors
        if self.errors:
            errors_df = pd.DataFrame(self.errors)
            errors_df.to_csv(self.log_dir / f'errors_{self.session_timestamp}.csv', index=False)

# Global trading logger instance
trading_logger = TradingLogger()