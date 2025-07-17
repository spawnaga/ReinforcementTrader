"""
Pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(4000, 4100, 100),
        'high': np.random.uniform(4050, 4150, 100),
        'low': np.random.uniform(3950, 4050, 100),
        'close': np.random.uniform(4000, 4100, 100),
        'volume': np.random.randint(1000, 10000, 100)
    })
    # Ensure high >= open, close, low
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    return data

@pytest.fixture
def temp_data_file(sample_market_data):
    """Create a temporary CSV file with market data"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_market_data.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)

@pytest.fixture
def mock_config():
    """Create a mock configuration object"""
    class MockConfig:
        def __init__(self):
            self.RISK_MANAGER_MAX_POSITION_SIZE = 5
            self.RISK_MANAGER_STOP_LOSS_PERCENT = 2.0
            self.RISK_MANAGER_TAKE_PROFIT_PERCENT = 4.0
            self.RISK_MANAGER_MAX_DAILY_LOSS = 1000
            self.RISK_MANAGER_MAX_TRADES_PER_DAY = 10
    
    return MockConfig()

@pytest.fixture
def sample_position():
    """Create a sample trading position"""
    return {
        'position_type': 'long',
        'entry_price': 4000.0,
        'position_size': 2,
        'entry_time': datetime.now()
    }