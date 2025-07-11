import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from pathlib import Path
import yfinance as yf
from extensions import db
from models import MarketData

logger = logging.getLogger(__name__)

class DataManager:
    """
    Advanced data management system for financial time series data
    Handles data loading, preprocessing, and feature engineering
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # NQ futures specifications
        self.nq_specs = {
            'symbol': 'NQ',
            'tick_size': 0.25,
            'value_per_tick': 5.0,
            'contract_size': 20,
            'trading_hours': {
                'start': '17:00',
                'end': '16:00'
            }
        }
        
        logger.info(f"Data Manager initialized with data directory: {self.data_dir}")
    
    def load_nq_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load NQ futures data from database or file
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # First try to load from database
            data = self._load_from_database('NQ', start_date, end_date)
            
            if data is not None and len(data) > 0:
                logger.info(f"Loaded {len(data)} NQ data points from database")
                return data
            
            # If no data in database, try to load from file
            data = self._load_from_file('NQ')
            
            if data is not None and len(data) > 0:
                # Save to database for future use
                self._save_to_database(data, 'NQ')
                logger.info(f"Loaded {len(data)} NQ data points from file and saved to database")
                return data
            
            # If no file data, try to download from Yahoo Finance (as proxy)
            data = self._download_market_data('NQ=F', start_date, end_date)
            
            if data is not None and len(data) > 0:
                # Save to database
                self._save_to_database(data, 'NQ')
                logger.info(f"Downloaded {len(data)} NQ data points and saved to database")
                return data
            
            logger.warning("No NQ data available from any source")
            return None
            
        except Exception as e:
            logger.error(f"Error loading NQ data: {str(e)}")
            return None
    
    def _load_from_database(self, symbol: str, start_date: Optional[str] = None, 
                           end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load data from database"""
        from app import app
        try:
            with app.app_context():
                query = MarketData.query.filter_by(symbol=symbol)
                
                if start_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    query = query.filter(MarketData.timestamp >= start_dt)
                
                if end_date:
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    query = query.filter(MarketData.timestamp <= end_dt)
                
                data = query.order_by(MarketData.timestamp.asc()).all()
                
                if not data:
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame([{
                    'timestamp': row.timestamp,
                    'open': row.open_price,
                    'high': row.high_price,
                    'low': row.low_price,
                    'close': row.close_price,
                    'volume': row.volume
                } for row in data])
                
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            return None
    
    def _load_from_file(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data from CSV file"""
        try:
            # Look for various file formats
            possible_files = [
                f"{symbol}_1min.csv",
                f"{symbol}_data.csv",
                f"{symbol}.csv",
                "nq_data.csv",
                "market_data.csv"
            ]
            
            for filename in possible_files:
                file_path = self.data_dir / filename
                if file_path.exists():
                    logger.info(f"Loading data from {file_path}")
                    
                    # Try different CSV formats
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Standardize column names
                        df.columns = df.columns.str.lower().str.strip()
                        
                        # Map common column name variations
                        column_mapping = {
                            'time': 'timestamp',
                            'datetime': 'timestamp',
                            'date': 'timestamp',
                            'o': 'open',
                            'h': 'high',
                            'l': 'low',
                            'c': 'close',
                            'v': 'volume',
                            'vol': 'volume'
                        }
                        
                        df.rename(columns=column_mapping, inplace=True)
                        
                        # Ensure required columns exist
                        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                        if not all(col in df.columns for col in required_cols):
                            logger.warning(f"Missing required columns in {filename}")
                            continue
                        
                        # Convert timestamp
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                        
                        # Sort by timestamp
                        df.sort_index(inplace=True)
                        
                        # Remove duplicates
                        df = df[~df.index.duplicated(keep='first')]
                        
                        # Basic data validation
                        if len(df) < 100:
                            logger.warning(f"Insufficient data in {filename}: {len(df)} rows")
                            continue
                        
                        return df
                        
                    except Exception as e:
                        logger.warning(f"Error reading {filename}: {str(e)}")
                        continue
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading data from file: {str(e)}")
            return None
    
    def _download_market_data(self, symbol: str, start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Download market data from Yahoo Finance"""
        try:
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"Downloading {symbol} data from {start_date} to {end_date}")
            
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval='1m')
            
            if data.empty:
                logger.warning(f"No data downloaded for {symbol}")
                return None
            
            # Standardize column names
            data.columns = data.columns.str.lower()
            data.reset_index(inplace=True)
            
            # Rename columns to match our standard
            data.rename(columns={
                'datetime': 'timestamp',
                'adj close': 'close'
            }, inplace=True)
            
            # Keep only required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            data = data[required_cols]
            
            # Set timestamp as index
            data.set_index('timestamp', inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading market data: {str(e)}")
            return None
    
    def _save_to_database(self, data: pd.DataFrame, symbol: str):
        """Save data to database"""
        from app import app
        try:
            with app.app_context():
                for timestamp, row in data.iterrows():
                    # Check if record already exists
                    existing = MarketData.query.filter_by(
                        symbol=symbol,
                        timestamp=timestamp
                    ).first()
                    
                    if existing:
                        continue
                    
                    # Create new record
                    market_data = MarketData(
                        timestamp=timestamp,
                        symbol=symbol,
                        open_price=float(row['open']),
                        high_price=float(row['high']),
                        low_price=float(row['low']),
                        close_price=float(row['close']),
                        volume=int(row['volume']) if not pd.isna(row['volume']) else 0,
                        timeframe='1min'
                    )
                    
                    db.session.add(market_data)
                
                db.session.commit()
                logger.info(f"Saved {len(data)} records to database for {symbol}")
            
        except Exception as e:
            logger.error(f"Error saving data to database: {str(e)}")
            if app.app_context:
                db.session.rollback()
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data with advanced feature engineering
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Preprocessed data with technical indicators
        """
        try:
            df = data.copy()
            
            # Basic price features
            df['price_range'] = df['high'] - df['low']
            df['price_change'] = df['close'] - df['open']
            df['price_change_pct'] = df['price_change'] / df['open'] * 100
            
            # Moving averages
            for window in [5, 10, 20, 50, 100]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            
            # Volatility measures
            df['volatility_5'] = df['close'].rolling(window=5).std()
            df['volatility_20'] = df['close'].rolling(window=20).std()
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = df['bb_upper'] - df['bb_lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_price_trend'] = df['volume'] * df['price_change']
            
            # Price patterns
            df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
            df['hammer'] = ((df['low'] < df[['open', 'close']].min(axis=1)) & 
                           (df['high'] - df[['open', 'close']].max(axis=1) < (df[['open', 'close']].max(axis=1) - df['low']) * 0.3)).astype(int)
            
            # Market structure
            df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
            df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
            
            # Time-based features
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            df['is_session_start'] = (df['hour'] == 17).astype(int)
            df['is_session_end'] = (df['hour'] == 16).astype(int)
            
            # Regime detection features
            df['trend_strength'] = df['close'].rolling(window=20).apply(
                lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
            )
            
            # Microstructure features
            df['tick_direction'] = np.where(df['close'] > df['close'].shift(1), 1, 
                                          np.where(df['close'] < df['close'].shift(1), -1, 0))
            df['uptick_ratio'] = df['tick_direction'].rolling(window=10).apply(lambda x: (x == 1).sum() / len(x))
            
            # Forward fill NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            logger.info(f"Preprocessed data with {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return data
    
    def get_market_hours_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter data to trading hours only"""
        try:
            # NQ futures trade nearly 24/7, but we can filter for active hours
            active_hours = data.between_time('17:00', '16:00')
            return active_hours
            
        except Exception as e:
            logger.error(f"Error filtering market hours: {str(e)}")
            return data
    
    def split_data(self, data: pd.DataFrame, train_ratio: float = 0.8, 
                   val_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets
        
        Args:
            data: Preprocessed data
            train_ratio: Ratio for training data
            val_ratio: Ratio for validation data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        try:
            n = len(data)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_data = data.iloc[:train_end]
            val_data = data.iloc[train_end:val_end]
            test_data = data.iloc[val_end:]
            
            logger.info(f"Split data: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return data, pd.DataFrame(), pd.DataFrame()
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict:
        """Get comprehensive data statistics"""
        try:
            stats = {
                'total_records': len(data),
                'date_range': {
                    'start': data.index.min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': data.index.max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'price_stats': {
                    'min_price': data['close'].min(),
                    'max_price': data['close'].max(),
                    'avg_price': data['close'].mean(),
                    'price_std': data['close'].std()
                },
                'volume_stats': {
                    'avg_volume': data['volume'].mean(),
                    'max_volume': data['volume'].max(),
                    'volume_std': data['volume'].std()
                },
                'missing_data': data.isnull().sum().to_dict(),
                'data_quality': {
                    'completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                    'consistency': len(data.dropna()) / len(data) * 100
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating data statistics: {str(e)}")
            return {}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        try:
            if data.empty:
                logger.error("Data is empty")
                return False
            
            # Check for required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for invalid price data
            if (data['high'] < data['low']).any():
                logger.error("Invalid price data: high < low")
                return False
            
            if (data['high'] < data[['open', 'close']].max(axis=1)).any():
                logger.error("Invalid price data: high < max(open, close)")
                return False
            
            if (data['low'] > data[['open', 'close']].min(axis=1)).any():
                logger.error("Invalid price data: low > min(open, close)")
                return False
            
            # Check for negative prices
            if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
                logger.error("Invalid price data: negative or zero prices")
                return False
            
            # Check for negative volume
            if (data['volume'] < 0).any():
                logger.error("Invalid volume data: negative volume")
                return False
            
            logger.info("Data validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
