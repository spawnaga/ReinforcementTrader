"""
Tests for data manager
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from data_manager import DataManager

class TestDataManager:
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager instance with temp directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DataManager(data_dir=tmpdir)
            yield dm
    
    def test_initialization(self, data_manager):
        """Test DataManager initialization"""
        assert data_manager is not None
        assert hasattr(data_manager, 'data_dir')
        assert hasattr(data_manager, 'gpu_loader')
    
    def test_load_data_from_csv(self, data_manager, temp_data_file):
        """Test loading data from CSV file"""
        df = data_manager.load_data_from_file(temp_data_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    
    def test_load_data_invalid_file(self, data_manager):
        """Test loading data from non-existent file"""
        with pytest.raises(Exception):
            data_manager.load_data_from_file("nonexistent.csv")
    
    def test_prepare_features(self, data_manager, sample_market_data):
        """Test feature preparation"""
        features = data_manager.prepare_features(sample_market_data)
        
        assert isinstance(features, pd.DataFrame)
        # Should have more columns than original due to technical indicators
        assert len(features.columns) > len(sample_market_data.columns)
        
        # Check for some expected technical indicators
        expected_indicators = ['sma_20', 'rsi_14', 'macd']
        for indicator in expected_indicators:
            assert any(indicator in col for col in features.columns)
    
    def test_create_sequences(self, data_manager, sample_market_data):
        """Test sequence creation for time series"""
        sequence_length = 10
        features = data_manager.prepare_features(sample_market_data)
        sequences, targets = data_manager.create_sequences(features, sequence_length)
        
        assert isinstance(sequences, np.ndarray)
        assert isinstance(targets, np.ndarray)
        
        # Check shapes
        expected_samples = len(features) - sequence_length
        assert sequences.shape[0] == expected_samples
        assert sequences.shape[1] == sequence_length
        assert sequences.shape[2] == features.shape[1]
        assert targets.shape[0] == expected_samples
    
    def test_normalize_data(self, data_manager, sample_market_data):
        """Test data normalization"""
        normalized = data_manager.normalize_data(sample_market_data[['open', 'high', 'low', 'close']])
        
        assert isinstance(normalized, pd.DataFrame)
        assert normalized.shape == sample_market_data[['open', 'high', 'low', 'close']].shape
        
        # Check that data is normalized (roughly between -3 and 3 for StandardScaler)
        assert normalized.abs().max().max() < 10
        
        # Mean should be close to 0, std close to 1
        assert abs(normalized.mean().mean()) < 0.1
        assert abs(normalized.std().mean() - 1.0) < 0.1
    
    def test_split_data(self, data_manager, sample_market_data):
        """Test train/test data split"""
        train_ratio = 0.8
        train_data, test_data = data_manager.split_data(sample_market_data, train_ratio)
        
        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)
        
        # Check sizes
        assert len(train_data) == int(len(sample_market_data) * train_ratio)
        assert len(test_data) == len(sample_market_data) - len(train_data)
        
        # Check no overlap
        assert train_data.index[-1] < test_data.index[0]
    
    def test_cache_data(self, data_manager, sample_market_data):
        """Test data caching functionality"""
        cache_key = "test_data"
        
        # Cache the data
        data_manager.cache_data(cache_key, sample_market_data)
        
        # Load from cache
        loaded_data = data_manager.load_cached_data(cache_key)
        
        assert loaded_data is not None
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == len(sample_market_data)
        pd.testing.assert_frame_equal(loaded_data, sample_market_data)
    
    def test_load_nonexistent_cache(self, data_manager):
        """Test loading non-existent cached data"""
        loaded_data = data_manager.load_cached_data("nonexistent_key")
        assert loaded_data is None
    
    def test_validate_data(self, data_manager, sample_market_data):
        """Test data validation"""
        # Valid data should pass
        is_valid = data_manager.validate_data(sample_market_data)
        assert is_valid == True
        
        # Test with missing columns
        invalid_data = sample_market_data.drop(columns=['volume'])
        is_valid = data_manager.validate_data(invalid_data)
        assert is_valid == False
        
        # Test with NaN values
        invalid_data = sample_market_data.copy()
        invalid_data.loc[0, 'close'] = np.nan
        is_valid = data_manager.validate_data(invalid_data)
        assert is_valid == False
    
    def test_resample_data(self, data_manager, sample_market_data):
        """Test data resampling"""
        # Ensure timestamp is index
        data = sample_market_data.set_index('timestamp')
        
        # Resample to 15 minutes
        resampled = data_manager.resample_data(data, '15min')
        
        assert isinstance(resampled, pd.DataFrame)
        assert len(resampled) < len(data)  # Should have fewer rows
        
        # Check OHLC logic
        # High should be max of highs
        # Low should be min of lows
        # Volume should be sum
        assert resampled['volume'].iloc[0] >= data['volume'].iloc[0]