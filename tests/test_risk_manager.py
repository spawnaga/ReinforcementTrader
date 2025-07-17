"""
Tests for risk manager
"""

import pytest
from datetime import datetime, timedelta
from risk_manager import RiskManager

class TestRiskManager:
    
    def test_initialization(self, mock_config):
        """Test risk manager initialization"""
        rm = RiskManager(mock_config)
        
        assert rm.max_position_size == 5
        assert rm.stop_loss_percent == 2.0
        assert rm.take_profit_percent == 4.0
        assert rm.max_daily_loss == 1000
        assert rm.max_trades_per_day == 10
    
    def test_check_position_size_valid(self, mock_config):
        """Test valid position size check"""
        rm = RiskManager(mock_config)
        
        assert rm.check_position_size(3) == True
        assert rm.check_position_size(5) == True
    
    def test_check_position_size_invalid(self, mock_config):
        """Test invalid position size check"""
        rm = RiskManager(mock_config)
        
        assert rm.check_position_size(6) == False
        assert rm.check_position_size(10) == False
        assert rm.check_position_size(0) == False
        assert rm.check_position_size(-1) == False
    
    def test_calculate_stop_loss(self, mock_config, sample_position):
        """Test stop loss calculation"""
        rm = RiskManager(mock_config)
        
        # Long position
        stop_loss = rm.calculate_stop_loss(
            sample_position['entry_price'],
            sample_position['position_type']
        )
        expected = 4000.0 * (1 - 0.02)  # 2% below entry
        assert abs(stop_loss - expected) < 0.01
        
        # Short position
        stop_loss = rm.calculate_stop_loss(4000.0, 'short')
        expected = 4000.0 * (1 + 0.02)  # 2% above entry
        assert abs(stop_loss - expected) < 0.01
    
    def test_calculate_take_profit(self, mock_config, sample_position):
        """Test take profit calculation"""
        rm = RiskManager(mock_config)
        
        # Long position
        take_profit = rm.calculate_take_profit(
            sample_position['entry_price'],
            sample_position['position_type']
        )
        expected = 4000.0 * (1 + 0.04)  # 4% above entry
        assert abs(take_profit - expected) < 0.01
        
        # Short position
        take_profit = rm.calculate_take_profit(4000.0, 'short')
        expected = 4000.0 * (1 - 0.04)  # 4% below entry
        assert abs(take_profit - expected) < 0.01
    
    def test_check_stop_loss_hit(self, mock_config):
        """Test stop loss hit detection"""
        rm = RiskManager(mock_config)
        
        # Long position scenarios
        position = {
            'position_type': 'long',
            'entry_price': 4000.0,
            'stop_loss': 3920.0  # 2% stop
        }
        
        assert rm.check_stop_loss(position, 3900.0) == True  # Hit
        assert rm.check_stop_loss(position, 3920.0) == True  # Hit at exact level
        assert rm.check_stop_loss(position, 3950.0) == False  # Not hit
        
        # Short position scenarios
        position = {
            'position_type': 'short',
            'entry_price': 4000.0,
            'stop_loss': 4080.0  # 2% stop
        }
        
        assert rm.check_stop_loss(position, 4100.0) == True  # Hit
        assert rm.check_stop_loss(position, 4080.0) == True  # Hit at exact level
        assert rm.check_stop_loss(position, 4050.0) == False  # Not hit
    
    def test_check_take_profit_hit(self, mock_config):
        """Test take profit hit detection"""
        rm = RiskManager(mock_config)
        
        # Long position scenarios
        position = {
            'position_type': 'long',
            'entry_price': 4000.0,
            'take_profit': 4160.0  # 4% target
        }
        
        assert rm.check_take_profit(position, 4200.0) == True  # Hit
        assert rm.check_take_profit(position, 4160.0) == True  # Hit at exact level
        assert rm.check_take_profit(position, 4100.0) == False  # Not hit
        
        # Short position scenarios
        position = {
            'position_type': 'short',
            'entry_price': 4000.0,
            'take_profit': 3840.0  # 4% target
        }
        
        assert rm.check_take_profit(position, 3800.0) == True  # Hit
        assert rm.check_take_profit(position, 3840.0) == True  # Hit at exact level
        assert rm.check_take_profit(position, 3900.0) == False  # Not hit
    
    def test_daily_loss_tracking(self, mock_config):
        """Test daily loss limit tracking"""
        rm = RiskManager(mock_config)
        
        # Add trades throughout the day
        rm.add_trade_result(-200)
        assert rm.check_daily_loss_limit() == True
        
        rm.add_trade_result(-300)
        assert rm.check_daily_loss_limit() == True
        assert rm.daily_pnl == -500
        
        rm.add_trade_result(-600)
        assert rm.check_daily_loss_limit() == False  # Exceeded limit
        assert rm.daily_pnl == -1100
    
    def test_daily_trade_count(self, mock_config):
        """Test daily trade count limit"""
        rm = RiskManager(mock_config)
        
        # Add trades
        for i in range(9):
            rm.add_trade_result(100)
            assert rm.check_trade_count_limit() == True
        
        # 10th trade should still be allowed
        rm.add_trade_result(100)
        assert rm.check_trade_count_limit() == True
        
        # 11th trade should be blocked
        assert rm.daily_trades == 10
        assert rm.check_trade_count_limit() == False
    
    def test_reset_daily_limits(self, mock_config):
        """Test daily limits reset"""
        rm = RiskManager(mock_config)
        
        # Add some trades
        rm.add_trade_result(-500)
        rm.add_trade_result(200)
        
        assert rm.daily_pnl == -300
        assert rm.daily_trades == 2
        
        # Reset
        rm.reset_daily_limits()
        
        assert rm.daily_pnl == 0
        assert rm.daily_trades == 0
        assert rm.last_reset_date == datetime.now().date()
    
    def test_position_size_calculation(self, mock_config):
        """Test position size calculation based on account equity"""
        rm = RiskManager(mock_config)
        
        # Test with different account sizes
        position_size = rm.calculate_position_size(100000, risk_percent=1.0)
        assert position_size == 1000  # 1% of 100k
        
        position_size = rm.calculate_position_size(50000, risk_percent=2.0)
        assert position_size == 1000  # 2% of 50k
        
        # Should respect max position size
        position_size = rm.calculate_position_size(
            1000000, 
            risk_percent=10.0,
            max_size=5000
        )
        assert position_size == 5000  # Capped at max