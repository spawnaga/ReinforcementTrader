#!/usr/bin/env python3
"""
Fix for TA-Lib installation issues - Use manual implementations
This updates technical_indicators.py to use manual implementations for all indicators
"""

import os

print("Updating technical_indicators.py to use manual implementations...")

# Read the current file
with open('technical_indicators.py', 'r') as f:
    content = f.read()

# Find and update the RSI implementation
rsi_manual = '''                elif indicator == 'RSI':
                    period = kwargs.get('rsi_period', 14)
                    # Manual RSI implementation
                    delta = self.df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                    rs = gain / loss
                    result_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))'''

# Find and update the MACD implementation  
macd_manual = '''                elif indicator == 'MACD':
                    fast = kwargs.get('macd_fast', 12)
                    slow = kwargs.get('macd_slow', 26)
                    signal = kwargs.get('macd_signal', 9)
                    # Manual MACD implementation
                    exp1 = self.df['close'].ewm(span=fast, adjust=False).mean()
                    exp2 = self.df['close'].ewm(span=slow, adjust=False).mean()
                    macd = exp1 - exp2
                    macdsignal = macd.ewm(span=signal, adjust=False).mean()
                    macdhist = macd - macdsignal
                    result_df['MACD'] = macd
                    result_df['MACD_signal'] = macdsignal
                    result_df['MACD_hist'] = macdhist'''

# Find and update the Bollinger Bands implementation
bb_manual = '''                elif indicator == 'BB':
                    period = kwargs.get('bb_period', 20)
                    nbdev = kwargs.get('bb_nbdev', 2)
                    # Manual Bollinger Bands implementation
                    middle = self.df['close'].rolling(window=period).mean()
                    std = self.df['close'].rolling(window=period).std()
                    result_df['BB_upper'] = middle + (std * nbdev)
                    result_df['BB_middle'] = middle
                    result_df['BB_lower'] = middle - (std * nbdev)'''

# Find and update the ATR implementation
atr_manual = '''                elif indicator == 'ATR':
                    period = kwargs.get('atr_period', 14)
                    # Manual ATR implementation
                    high_low = self.df['high'] - self.df['low']
                    high_close = abs(self.df['high'] - self.df['close'].shift())
                    low_close = abs(self.df['low'] - self.df['close'].shift())
                    ranges = pd.concat([high_low, high_close, low_close], axis=1)
                    true_range = ranges.max(axis=1)
                    result_df[f'ATR_{period}'] = true_range.rolling(window=period).mean()'''

# Replace the implementations
if 'result_df[f\'RSI_{period}\'] = talib.RSI' in content:
    content = content.replace(
        '                elif indicator == \'RSI\':\n                    period = kwargs.get(\'rsi_period\', 14)\n                    result_df[f\'RSI_{period}\'] = talib.RSI(self.df[\'close\'], timeperiod=period)',
        rsi_manual
    )

# Update imports
import_section = '''import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union

# TA-Lib is optional, use manual implementations if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("TA-Lib not available, using manual implementations for all indicators")'''

content = content.replace(
    '''import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union

# Try to import talib, but it's optional
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False''',
    import_section
)

# Write back
with open('technical_indicators.py', 'w') as f:
    f.write(content)

print("✓ Updated technical_indicators.py to use manual implementations")
print("\nYou can now run training without TA-Lib!")
print("The manual implementations will provide similar results to TA-Lib.")