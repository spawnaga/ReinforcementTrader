"""
Futures Contract Specifications
Defines specifications for different futures contracts (ES, NQ, etc.)
"""

from typing import Dict, Any

class FuturesContract:
    """Base class for futures contract specifications"""
    
    def __init__(self, symbol: str, specs: Dict[str, Any]):
        self.symbol = symbol
        self.name = specs.get('name', symbol)
        self.exchange = specs.get('exchange', 'CME')
        self.tick_size = specs.get('tick_size', 0.25)
        self.value_per_tick = specs.get('value_per_tick', 5.0)
        self.multiplier = specs.get('multiplier', 1.0)
        self.currency = specs.get('currency', 'USD')
        self.trading_hours = specs.get('trading_hours', '24/5')
        self.margin_requirements = specs.get('margin_requirements', {})
        self.min_holding_periods = specs.get('min_holding_periods', 5)
        self.execution_cost_per_order = specs.get('execution_cost_per_order', 5.0)
        self.slippage_ticks = specs.get('slippage_ticks', 1)
        
    def calculate_value(self, ticks: float) -> float:
        """Calculate dollar value from number of ticks"""
        return ticks * self.value_per_tick
        
    def calculate_ticks(self, price_change: float) -> float:
        """Calculate number of ticks from price change"""
        return price_change / self.tick_size
        

# Contract specifications for major futures
FUTURES_SPECS = {
    'NQ': {
        'name': 'E-mini Nasdaq 100',
        'exchange': 'CME',
        'tick_size': 0.25,
        'value_per_tick': 5.0,  # $5 per tick
        'multiplier': 20.0,     # $20 per point
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 17600,    # Approximate initial margin
            'maintenance': 16000  # Approximate maintenance margin
        },
        'min_holding_periods': 10,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 2
    },
    
    'ES': {
        'name': 'E-mini S&P 500',
        'exchange': 'CME',
        'tick_size': 0.25,
        'value_per_tick': 12.50,  # $12.50 per tick
        'multiplier': 50.0,        # $50 per point
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 13200,      # Approximate initial margin
            'maintenance': 12000    # Approximate maintenance margin
        },
        'min_holding_periods': 10,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 1
    },
    
    'YM': {
        'name': 'E-mini Dow Jones',
        'exchange': 'CBOT',
        'tick_size': 1.0,
        'value_per_tick': 5.0,   # $5 per tick
        'multiplier': 5.0,       # $5 per point
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 8800,     # Approximate initial margin
            'maintenance': 8000   # Approximate maintenance margin
        },
        'min_holding_periods': 10,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 1
    },
    
    'RTY': {
        'name': 'E-mini Russell 2000',
        'exchange': 'CME',
        'tick_size': 0.10,
        'value_per_tick': 5.0,   # $5 per tick
        'multiplier': 50.0,      # $50 per point
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 7700,     # Approximate initial margin
            'maintenance': 7000   # Approximate maintenance margin
        },
        'min_holding_periods': 10,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 2
    },
    
    'CL': {
        'name': 'Crude Oil',
        'exchange': 'NYMEX',
        'tick_size': 0.01,
        'value_per_tick': 10.0,  # $10 per tick
        'multiplier': 1000.0,    # 1000 barrels
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 6600,     # Approximate initial margin
            'maintenance': 6000   # Approximate maintenance margin
        },
        'min_holding_periods': 15,
        'execution_cost_per_order': 7.0,
        'slippage_ticks': 2
    },
    
    'GC': {
        'name': 'Gold',
        'exchange': 'COMEX',
        'tick_size': 0.10,
        'value_per_tick': 10.0,  # $10 per tick
        'multiplier': 100.0,     # 100 troy ounces
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 11000,    # Approximate initial margin
            'maintenance': 10000  # Approximate maintenance margin
        },
        'min_holding_periods': 20,
        'execution_cost_per_order': 7.0,
        'slippage_ticks': 1
    },
    
    'ZN': {
        'name': '10-Year T-Note',
        'exchange': 'CBOT',
        'tick_size': 0.015625,   # 1/64
        'value_per_tick': 15.625, # $15.625 per tick
        'multiplier': 1000.0,     # $1000 per point
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 1650,      # Approximate initial margin
            'maintenance': 1500    # Approximate maintenance margin
        },
        'min_holding_periods': 30,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 1
    },
    
    '6E': {
        'name': 'Euro FX',
        'exchange': 'CME',
        'tick_size': 0.00005,
        'value_per_tick': 6.25,  # $6.25 per tick
        'multiplier': 125000.0,  # 125,000 euros
        'currency': 'USD',
        'trading_hours': 'Nearly 24/5',
        'margin_requirements': {
            'initial': 2310,     # Approximate initial margin
            'maintenance': 2100   # Approximate maintenance margin
        },
        'min_holding_periods': 20,
        'execution_cost_per_order': 5.0,
        'slippage_ticks': 1
    }
}


def get_contract(symbol: str) -> FuturesContract:
    """
    Get futures contract specifications
    
    Args:
        symbol: Contract symbol (e.g., 'NQ', 'ES')
        
    Returns:
        FuturesContract object with specifications
    """
    symbol = symbol.upper()
    if symbol not in FUTURES_SPECS:
        # Default to NQ specifications with warning
        import logging
        logging.warning(f"Unknown futures contract {symbol}, using NQ defaults")
        return FuturesContract(symbol, FUTURES_SPECS['NQ'])
    
    return FuturesContract(symbol, FUTURES_SPECS[symbol])


def list_available_contracts() -> Dict[str, str]:
    """List all available futures contracts"""
    return {symbol: specs['name'] for symbol, specs in FUTURES_SPECS.items()}