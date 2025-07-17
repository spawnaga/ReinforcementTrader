"""
Technical Indicators Module
Provides a comprehensive set of technical indicators for trading analysis
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import talib, fallback to manual implementations if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("TA-Lib not available, using manual implementations")

# Constants for time-based indicators
SECONDS_IN_DAY = 24 * 60 * 60
MINUTES_IN_DAY = 24 * 60
WEEKDAYS = 7

class TechnicalIndicators:
    """
    Technical indicators calculator with abbreviations and descriptions
    """
    
    # Indicator registry with abbreviations and descriptions
    INDICATORS = {
        # Time-based indicators
        'sin_time': 'Sine of time of day (cyclical feature)',
        'cos_time': 'Cosine of time of day (cyclical feature)',
        'sin_weekday': 'Sine of day of week (cyclical feature)', 
        'cos_weekday': 'Cosine of day of week (cyclical feature)',
        'sin_hour': 'Sine of hour (cyclical feature)',
        'cos_hour': 'Cosine of hour (cyclical feature)',
        'hour': 'Hour of day (0-23)',
        'minute': 'Minute of hour (0-59)',
        'day_of_week': 'Day of week (0-6, Monday=0)',
        
        # Moving averages
        'SMA': 'Simple Moving Average',
        'EMA': 'Exponential Moving Average',
        'WMA': 'Weighted Moving Average',
        'DEMA': 'Double Exponential Moving Average',
        'TEMA': 'Triple Exponential Moving Average',
        
        # Momentum indicators
        'RSI': 'Relative Strength Index',
        'MACD': 'Moving Average Convergence Divergence',
        'STOCH': 'Stochastic Oscillator',
        'CCI': 'Commodity Channel Index',
        'MOM': 'Momentum',
        'ROC': 'Rate of Change',
        'WILLR': 'Williams %R',
        'MFI': 'Money Flow Index',
        
        # Volatility indicators
        'ATR': 'Average True Range',
        'NATR': 'Normalized Average True Range',
        'BB': 'Bollinger Bands',
        'KC': 'Keltner Channels',
        'DC': 'Donchian Channels',
        
        # Volume indicators
        'OBV': 'On Balance Volume',
        'AD': 'Accumulation/Distribution',
        'ADOSC': 'Chaikin A/D Oscillator',
        'CMF': 'Chaikin Money Flow',
        
        # Trend indicators
        'ADX': 'Average Directional Index',
        'AROON': 'Aroon Indicator',
        'PSAR': 'Parabolic SAR',
        'TRIX': 'Triple Exponential Average',
        
        # Pattern recognition
        'CDL': 'Candlestick Patterns',
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV dataframe
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        """
        self.df = df.copy()
        self._prepare_time_features()
        
    def _prepare_time_features(self):
        """Prepare time-based features from timestamp"""
        if 'timestamp' in self.df.columns:
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Extract time components
            self.df['hour'] = self.df['timestamp'].dt.hour
            self.df['minute'] = self.df['timestamp'].dt.minute
            self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
            
            # Calculate minutes since midnight
            minutes_since_midnight = self.df['hour'] * 60 + self.df['minute']
            
            # Cyclical time features
            self.df['sin_time'] = np.sin(2 * np.pi * minutes_since_midnight / MINUTES_IN_DAY)
            self.df['cos_time'] = np.cos(2 * np.pi * minutes_since_midnight / MINUTES_IN_DAY)
            
            # Cyclical weekday features
            self.df['sin_weekday'] = np.sin(2 * np.pi * self.df['day_of_week'] / WEEKDAYS)
            self.df['cos_weekday'] = np.cos(2 * np.pi * self.df['day_of_week'] / WEEKDAYS)
            
    def calculate_indicators(self, indicators: List[str], **kwargs) -> pd.DataFrame:
        """
        Calculate specified indicators
        
        Args:
            indicators: List of indicator abbreviations to calculate
            **kwargs: Additional parameters for indicators (e.g., period=14)
            
        Returns:
            DataFrame with calculated indicators
        """
        result_df = self.df.copy()
        
        for indicator in indicators:
            if indicator in ['sin_time', 'cos_time', 'sin_weekday', 'cos_weekday', 
                           'hour', 'minute', 'day_of_week']:
                # Time-based indicators already calculated
                continue
                
            try:
                if indicator == 'SMA':
                    period = kwargs.get('sma_period', 20)
                    if HAS_TALIB:
                        result_df[f'SMA_{period}'] = talib.SMA(self.df['close'], timeperiod=period)
                    else:
                        result_df[f'SMA_{period}'] = self.df['close'].rolling(window=period).mean()
                    
                elif indicator == 'EMA':
                    period = kwargs.get('ema_period', 20)
                    if HAS_TALIB:
                        result_df[f'EMA_{period}'] = talib.EMA(self.df['close'], timeperiod=period)
                    else:
                        result_df[f'EMA_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
                    
                elif indicator == 'RSI':
                    period = kwargs.get('rsi_period', 14)
                    result_df[f'RSI_{period}'] = talib.RSI(self.df['close'], timeperiod=period)
                    
                elif indicator == 'MACD':
                    fast = kwargs.get('macd_fast', 12)
                    slow = kwargs.get('macd_slow', 26)
                    signal = kwargs.get('macd_signal', 9)
                    macd, macdsignal, macdhist = talib.MACD(self.df['close'], 
                                                           fastperiod=fast, 
                                                           slowperiod=slow, 
                                                           signalperiod=signal)
                    result_df['MACD'] = macd
                    result_df['MACD_signal'] = macdsignal
                    result_df['MACD_hist'] = macdhist
                    
                elif indicator == 'BB':
                    period = kwargs.get('bb_period', 20)
                    nbdev = kwargs.get('bb_nbdev', 2)
                    upper, middle, lower = talib.BBANDS(self.df['close'], 
                                                       timeperiod=period, 
                                                       nbdevup=nbdev, 
                                                       nbdevdn=nbdev)
                    result_df['BB_upper'] = upper
                    result_df['BB_middle'] = middle
                    result_df['BB_lower'] = lower
                    
                elif indicator == 'ATR':
                    period = kwargs.get('atr_period', 14)
                    result_df[f'ATR_{period}'] = talib.ATR(self.df['high'], 
                                                          self.df['low'], 
                                                          self.df['close'], 
                                                          timeperiod=period)
                    
                elif indicator == 'STOCH':
                    fastk_period = kwargs.get('stoch_fastk', 14)
                    slowk_period = kwargs.get('stoch_slowk', 3)
                    slowd_period = kwargs.get('stoch_slowd', 3)
                    slowk, slowd = talib.STOCH(self.df['high'], 
                                              self.df['low'], 
                                              self.df['close'],
                                              fastk_period=fastk_period,
                                              slowk_period=slowk_period,
                                              slowd_period=slowd_period)
                    result_df['STOCH_K'] = slowk
                    result_df['STOCH_D'] = slowd
                    
                elif indicator == 'ADX':
                    period = kwargs.get('adx_period', 14)
                    result_df[f'ADX_{period}'] = talib.ADX(self.df['high'], 
                                                          self.df['low'], 
                                                          self.df['close'], 
                                                          timeperiod=period)
                    
                elif indicator == 'CCI':
                    period = kwargs.get('cci_period', 20)
                    result_df[f'CCI_{period}'] = talib.CCI(self.df['high'], 
                                                          self.df['low'], 
                                                          self.df['close'], 
                                                          timeperiod=period)
                    
                elif indicator == 'OBV':
                    result_df['OBV'] = talib.OBV(self.df['close'], self.df['volume'])
                    
                elif indicator == 'MFI':
                    period = kwargs.get('mfi_period', 14)
                    result_df[f'MFI_{period}'] = talib.MFI(self.df['high'], 
                                                          self.df['low'], 
                                                          self.df['close'], 
                                                          self.df['volume'],
                                                          timeperiod=period)
                    
                else:
                    logger.warning(f"Indicator {indicator} not implemented yet")
                    
            except Exception as e:
                logger.error(f"Error calculating {indicator}: {e}")
                
        return result_df
    
    @staticmethod
    def get_indicator_info(indicator: str) -> Dict[str, str]:
        """Get information about an indicator"""
        return {
            'abbreviation': indicator,
            'description': TechnicalIndicators.INDICATORS.get(indicator, 'Unknown indicator'),
            'available': indicator in TechnicalIndicators.INDICATORS
        }
    
    @staticmethod
    def list_all_indicators() -> Dict[str, str]:
        """List all available indicators with descriptions"""
        return TechnicalIndicators.INDICATORS.copy()


def add_time_based_indicators(df: pd.DataFrame, use_cudf: bool = False) -> pd.DataFrame:
    """
    Add time-based indicators using sin/cos transformations
    Compatible with both pandas and cuDF
    
    Args:
        df: DataFrame with timestamp column
        use_cudf: Whether to use cuDF acceleration
        
    Returns:
        DataFrame with added time indicators
    """
    HAS_CUDF = False
    if use_cudf:
        try:
            import cudf
            HAS_CUDF = True
        except ImportError:
            logger.warning("cuDF not available, using pandas")
    
    # Extract time components
    if 'timestamp' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        hour = df['timestamp'].dt.hour
        minute = df['timestamp'].dt.minute
        weekday = df['timestamp'].dt.dayofweek
        
        # Calculate minutes since midnight
        minutes = hour * 60 + minute
        
        try:
            # Convert cuDF/pandas series to numpy arrays for consistent CPU processing
            if HAS_CUDF and hasattr(minutes, 'to_pandas'):
                minutes_np = minutes.to_pandas().values
                weekday_np = weekday.to_pandas().values
                hour_np = hour.to_pandas().values
            else:
                minutes_np = minutes.values if hasattr(minutes, 'values') else np.array(minutes)
                weekday_np = weekday.values if hasattr(weekday, 'values') else np.array(weekday)
                hour_np = hour.values if hasattr(hour, 'values') else np.array(hour)

            # Ensure arrays are valid
            minutes_np = np.nan_to_num(minutes_np, nan=0.0)
            weekday_np = np.nan_to_num(weekday_np, nan=0.0)
            hour_np = np.nan_to_num(hour_np, nan=0.0)

            # Perform trigonometric operations on CPU
            sin_time_vals = np.sin(2 * np.pi * (minutes_np / MINUTES_IN_DAY))
            cos_time_vals = np.cos(2 * np.pi * (minutes_np / MINUTES_IN_DAY))
            sin_weekday_vals = np.sin(2 * np.pi * (weekday_np / WEEKDAYS))
            cos_weekday_vals = np.cos(2 * np.pi * (weekday_np / WEEKDAYS))
            sin_hour_vals = np.sin(2 * np.pi * (hour_np / 24))
            cos_hour_vals = np.cos(2 * np.pi * (hour_np / 24))

            # Ensure no NaN values in trigonometric results
            sin_time_vals = np.nan_to_num(sin_time_vals, nan=0.0)
            cos_time_vals = np.nan_to_num(cos_time_vals, nan=1.0)
            sin_weekday_vals = np.nan_to_num(sin_weekday_vals, nan=0.0)
            cos_weekday_vals = np.nan_to_num(cos_weekday_vals, nan=1.0)
            sin_hour_vals = np.nan_to_num(sin_hour_vals, nan=0.0)
            cos_hour_vals = np.nan_to_num(cos_hour_vals, nan=1.0)

            # Assign back to dataframe
            df["sin_time"] = sin_time_vals
            df["cos_time"] = cos_time_vals
            df["sin_weekday"] = sin_weekday_vals
            df["cos_weekday"] = cos_weekday_vals
            df["sin_hour"] = sin_hour_vals
            df["cos_hour"] = cos_hour_vals

        except Exception as e:
            logger.warning(f"Trigonometric operations failed: {e}, using fallback")
            # Final fallback with basic values
            df["sin_time"] = 0.0
            df["cos_time"] = 1.0
            df["sin_weekday"] = 0.0
            df["cos_weekday"] = 1.0
            
    return df