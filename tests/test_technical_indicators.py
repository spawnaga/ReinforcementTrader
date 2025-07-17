"""
Tests for technical indicators
"""

import pytest
import pandas as pd
import numpy as np
from technical_indicators import TechnicalIndicators

class TestTechnicalIndicators:
    
    def test_sma(self, sample_market_data):
        """Test Simple Moving Average calculation"""
        ti = TechnicalIndicators(sample_market_data)
        result = ti.calculate_indicators(['SMA'], sma_period=20)
        
        assert len(result) == len(sample_market_data)
        assert result.iloc[:19].isna().all()  # First 19 should be NaN
        assert not result.iloc[19:].isna().any()  # Rest should have values
        
        # Check calculation for a specific point
        expected = sample_market_data['close'].iloc[0:20].mean()
        assert abs(result.iloc[19] - expected) < 0.01
    
    def test_ema(self, sample_market_data):
        """Test Exponential Moving Average calculation"""
        result = TechnicalIndicators.ema(sample_market_data['close'], period=20)
        
        assert len(result) == len(sample_market_data)
        assert not result.isna().any()  # EMA should have values for all points
        
        # EMA should be different from SMA
        sma = TechnicalIndicators.sma(sample_market_data['close'], period=20)
        assert not np.allclose(result[20:], sma[20:], rtol=0.1)
    
    def test_rsi(self, sample_market_data):
        """Test RSI calculation"""
        result = TechnicalIndicators.rsi(sample_market_data['close'], period=14)
        
        assert len(result) == len(sample_market_data)
        assert result.iloc[:14].isna().all()  # First 14 should be NaN
        
        # RSI should be between 0 and 100
        valid_rsi = result.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()
    
    def test_macd(self, sample_market_data):
        """Test MACD calculation"""
        macd, signal, histogram = TechnicalIndicators.macd(
            sample_market_data['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        
        assert len(macd) == len(sample_market_data)
        assert len(signal) == len(sample_market_data)
        assert len(histogram) == len(sample_market_data)
        
        # Histogram should be MACD - Signal
        valid_idx = ~(macd.isna() | signal.isna())
        expected_hist = macd[valid_idx] - signal[valid_idx]
        assert np.allclose(histogram[valid_idx], expected_hist)
    
    def test_bollinger_bands(self, sample_market_data):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            sample_market_data['close'],
            period=20,
            std_dev=2
        )
        
        assert len(upper) == len(sample_market_data)
        assert len(middle) == len(sample_market_data)
        assert len(lower) == len(sample_market_data)
        
        # Middle band should be SMA
        sma = TechnicalIndicators.sma(sample_market_data['close'], period=20)
        assert np.allclose(middle.dropna(), sma.dropna())
        
        # Upper should be above middle, lower should be below
        valid_idx = ~middle.isna()
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (lower[valid_idx] < middle[valid_idx]).all()
    
    def test_stochastic(self, sample_market_data):
        """Test Stochastic calculation"""
        k, d = TechnicalIndicators.stochastic(
            sample_market_data['high'],
            sample_market_data['low'],
            sample_market_data['close'],
            k_period=14,
            d_period=3
        )
        
        assert len(k) == len(sample_market_data)
        assert len(d) == len(sample_market_data)
        
        # Values should be between 0 and 100
        valid_k = k.dropna()
        valid_d = d.dropna()
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_atr(self, sample_market_data):
        """Test Average True Range calculation"""
        result = TechnicalIndicators.atr(
            sample_market_data['high'],
            sample_market_data['low'],
            sample_market_data['close'],
            period=14
        )
        
        assert len(result) == len(sample_market_data)
        
        # ATR should be positive
        valid_atr = result.dropna()
        assert (valid_atr > 0).all()
    
    def test_williams_r(self, sample_market_data):
        """Test Williams %R calculation"""
        result = TechnicalIndicators.williams_r(
            sample_market_data['high'],
            sample_market_data['low'],
            sample_market_data['close'],
            period=14
        )
        
        assert len(result) == len(sample_market_data)
        
        # Williams %R should be between -100 and 0
        valid_wr = result.dropna()
        assert (valid_wr >= -100).all() and (valid_wr <= 0).all()
    
    def test_vwap(self, sample_market_data):
        """Test VWAP calculation"""
        result = TechnicalIndicators.vwap(
            sample_market_data['high'],
            sample_market_data['low'],
            sample_market_data['close'],
            sample_market_data['volume']
        )
        
        assert len(result) == len(sample_market_data)
        
        # VWAP should be positive and within price range
        valid_vwap = result.dropna()
        assert (valid_vwap > 0).all()
        assert (valid_vwap >= sample_market_data['low'].min()).all()
        assert (valid_vwap <= sample_market_data['high'].max()).all()
    
    def test_add_all_indicators(self, sample_market_data):
        """Test adding all indicators to dataframe"""
        df = sample_market_data.copy()
        result = TechnicalIndicators.add_all_indicators(df)
        
        # Check that all indicator columns were added
        expected_columns = [
            'sma_20', 'ema_20', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d',
            'atr_14', 'williams_r', 'vwap'
        ]
        
        for col in expected_columns:
            assert col in result.columns
        
        # Original columns should still be there
        for col in sample_market_data.columns:
            assert col in result.columns