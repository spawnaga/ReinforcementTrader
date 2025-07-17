"""
Simplified tests for technical indicators
"""

import pytest
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators

class TestTechnicalIndicatorsSimple:
    
    def test_initialization(self, sample_market_data):
        """Test TechnicalIndicators initialization"""
        ti = TechnicalIndicators(sample_market_data)
        
        assert ti is not None
        assert isinstance(ti.df, pd.DataFrame)
        assert len(ti.df) == len(sample_market_data)
        
        # Check if time features were added
        if 'timestamp' in sample_market_data.columns:
            assert 'hour' in ti.df.columns
            assert 'minute' in ti.df.columns
            assert 'day_of_week' in ti.df.columns
    
    def test_calculate_sma(self, sample_market_data):
        """Test SMA calculation"""
        ti = TechnicalIndicators(sample_market_data)
        result = ti.calculate_indicators(['SMA'], sma_period=20)
        
        assert 'SMA_20' in result.columns
        # First 19 values should be NaN
        assert result['SMA_20'].iloc[:19].isna().all()
        # Rest should have values
        assert not result['SMA_20'].iloc[20:].isna().any()
    
    def test_calculate_ema(self, sample_market_data):
        """Test EMA calculation"""
        ti = TechnicalIndicators(sample_market_data)
        result = ti.calculate_indicators(['EMA'], ema_period=20)
        
        assert 'EMA_20' in result.columns
        # EMA should have values throughout
        assert not result['EMA_20'].isna().all()
    
    def test_multiple_indicators(self, sample_market_data):
        """Test calculating multiple indicators"""
        ti = TechnicalIndicators(sample_market_data)
        result = ti.calculate_indicators(['SMA', 'EMA'], sma_period=10, ema_period=10)
        
        assert 'SMA_10' in result.columns
        assert 'EMA_10' in result.columns
    
    def test_get_indicator_info(self):
        """Test getting indicator information"""
        info = TechnicalIndicators.get_indicator_info('RSI')
        
        assert info['abbreviation'] == 'RSI'
        assert info['description'] == 'Relative Strength Index'
        assert info['available'] == True
    
    def test_list_all_indicators(self):
        """Test listing all indicators"""
        indicators = TechnicalIndicators.list_all_indicators()
        
        assert isinstance(indicators, dict)
        assert 'RSI' in indicators
        assert 'MACD' in indicators
        assert 'SMA' in indicators
        assert indicators['RSI'] == 'Relative Strength Index'