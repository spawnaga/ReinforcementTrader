"""
Tests for trading logger
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from trading_logger import TradingLogger

class TestTradingLogger:
    
    @pytest.fixture
    def trading_logger(self):
        """Create a TradingLogger instance with temp directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TradingLogger(session_id=1, log_dir=tmpdir)
            yield logger
    
    def test_initialization(self, trading_logger):
        """Test TradingLogger initialization"""
        assert trading_logger.session_id == 1
        assert trading_logger.log_dir.exists()
        
        # Check that log files were created
        expected_files = [
            'trading.log',
            'positions.log',
            'rewards.log',
            'errors.log',
            'debug.log'
        ]
        
        for filename in expected_files:
            log_file = trading_logger.log_dir / filename
            assert log_file.exists()
    
    def test_log_trade(self, trading_logger):
        """Test logging a trade"""
        trading_logger.log_trade(
            timestamp=datetime.now(),
            action='buy',
            position_type='long',
            entry_price=4000.0,
            position_size=2,
            reason='Signal triggered'
        )
        
        # Check that trade was logged
        trading_log = trading_logger.log_dir / 'trading.log'
        content = trading_log.read_text()
        
        assert 'BUY' in content
        assert 'long' in content
        assert '4000.0' in content
        assert 'Signal triggered' in content
    
    def test_log_position_update(self, trading_logger):
        """Test logging position updates"""
        trading_logger.log_position_update(
            timestamp=datetime.now(),
            position_type='short',
            entry_price=4100.0,
            current_price=4080.0,
            position_size=1,
            unrealized_pnl=20.0,
            realized_pnl=0.0
        )
        
        # Check that position was logged
        positions_log = trading_logger.log_dir / 'positions.log'
        content = positions_log.read_text()
        
        assert 'short' in content
        assert '4100.0' in content
        assert '4080.0' in content
        assert '20.0' in content
    
    def test_log_reward(self, trading_logger):
        """Test logging rewards"""
        trading_logger.log_reward(
            timestamp=datetime.now(),
            episode=10,
            step=50,
            reward=15.5,
            total_reward=150.0,
            action='hold'
        )
        
        # Check that reward was logged
        rewards_log = trading_logger.log_dir / 'rewards.log'
        content = rewards_log.read_text()
        
        assert 'Episode: 10' in content
        assert 'Step: 50' in content
        assert '15.5' in content
        assert '150.0' in content
    
    def test_log_error(self, trading_logger):
        """Test logging errors"""
        trading_logger.log_error(
            timestamp=datetime.now(),
            error_type='OrderError',
            message='Insufficient margin',
            details={'required': 5000, 'available': 4000}
        )
        
        # Check that error was logged
        errors_log = trading_logger.log_dir / 'errors.log'
        content = errors_log.read_text()
        
        assert 'OrderError' in content
        assert 'Insufficient margin' in content
        assert 'required' in content
        assert '5000' in content
    
    def test_log_debug(self, trading_logger):
        """Test debug logging"""
        trading_logger.log_debug(
            timestamp=datetime.now(),
            message='State calculation',
            data={'rsi': 65.5, 'macd': 0.5}
        )
        
        # Check that debug info was logged
        debug_log = trading_logger.log_dir / 'debug.log'
        content = debug_log.read_text()
        
        assert 'State calculation' in content
        assert 'rsi' in content
        assert '65.5' in content
    
    def test_get_summary(self, trading_logger):
        """Test getting trading summary"""
        # Log some trades
        trading_logger.log_trade(
            timestamp=datetime.now(),
            action='buy',
            position_type='long',
            entry_price=4000.0,
            position_size=1
        )
        
        trading_logger.log_trade(
            timestamp=datetime.now(),
            action='sell',
            position_type='long',
            entry_price=4000.0,
            exit_price=4050.0,
            position_size=1,
            profit_loss=50.0
        )
        
        summary = trading_logger.get_summary()
        
        assert 'total_trades' in summary
        assert 'profitable_trades' in summary
        assert 'total_profit_loss' in summary
        assert summary['total_trades'] >= 2
    
    def test_export_logs(self, trading_logger):
        """Test exporting logs to single file"""
        # Add some logs
        trading_logger.log_trade(
            timestamp=datetime.now(),
            action='buy',
            position_type='long',
            entry_price=4000.0,
            position_size=1
        )
        
        trading_logger.log_error(
            timestamp=datetime.now(),
            error_type='TestError',
            message='Test error message'
        )
        
        # Export logs
        export_path = trading_logger.log_dir / 'exported_logs.txt'
        trading_logger.export_logs(export_path)
        
        assert export_path.exists()
        content = export_path.read_text()
        
        # Should contain content from multiple log files
        assert 'Trading Log' in content
        assert 'Error Log' in content
        assert 'BUY' in content
        assert 'TestError' in content