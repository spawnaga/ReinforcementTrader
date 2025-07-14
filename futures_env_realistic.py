"""
Realistic Futures Trading Environment - Fixed Version
This fixes the instant trading exploit by adding:
1. Minimum holding periods
2. Realistic transaction costs
3. Slippage simulation
4. Trade frequency limits
"""

import datetime
import logging
from pathlib import Path
from typing import List, Tuple, Sequence, Optional
from uuid import uuid4

import gym
import numpy as np
from gym import spaces

from gym_futures.envs.utils import round_to_nearest_increment, TimeSeriesState

logger = logging.getLogger(__name__)

# Setup detailed trading log file
import os
os.makedirs('logs', exist_ok=True)
trading_logger = logging.getLogger('trading')
trading_handler = logging.FileHandler('logs/trading.log')
trading_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
trading_logger.addHandler(trading_handler)
trading_logger.setLevel(logging.DEBUG)


def calculate_reward(timestamp, action, position_before, position_after, entry_price, exit_price, position_type,
                     value_per_tick, execution_cost, session_id, tick_size):
    """Calculate the reward for a trading action with realistic costs"""
    if entry_price is None or exit_price is None:
        return 0.0

    if action == 'EXIT':
        if position_type == 'long':
            profit_ticks = (exit_price - entry_price) / tick_size
        elif position_type == 'short':
            profit_ticks = (entry_price - exit_price) / tick_size
        else:
            return 0.0
        
        # Higher execution costs for realistic trading
        net_profit = profit_ticks * value_per_tick - execution_cost
        
        trading_logger.debug(
            f"REWARD CALC: timestamp={timestamp}, action={action}, position_type={position_type}, "
            f"entry={entry_price}, exit={exit_price}, profit_ticks={profit_ticks}, net_profit={net_profit}"
        )
        return net_profit
    
    elif action == 'HOLD' and position_after != 0:
        # Small reward for holding positions (encourages patience)
        if position_type == 'long':
            unrealized_ticks = (exit_price - entry_price) / tick_size
        elif position_type == 'short':
            unrealized_ticks = (entry_price - exit_price) / tick_size
        else:
            return 0.0
        unrealized = unrealized_ticks * value_per_tick
        return unrealized * 0.01  # Very small reward for holding
    else:
        return 0.0


class RealisticFuturesEnv(gym.Env):
    """
    A realistic futures trading environment that prevents instant trading exploits.
    
    Key features:
    - Minimum holding period enforcement
    - Realistic transaction costs and slippage
    - Trade frequency limits
    - Time-based restrictions
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 states: Sequence[TimeSeriesState], 
                 value_per_tick: float,
                 tick_size: float, 
                 fill_probability: float = 0.95,  # Not all trades fill
                 execution_cost_per_order: float = 5.0,  # $5 per side
                 min_holding_periods: int = 10,  # Must hold for 10 time steps
                 max_trades_per_episode: int = 5,  # Max 5 trades per episode
                 slippage_ticks: int = 2,  # Average 2 tick slippage
                 add_current_position_to_state: bool = True,
                 session_id: Optional[int] = None,
                 enable_trading_logger: bool = True):
        
        self.states = states
        self.limit = len(self.states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.add_current_position_to_state = add_current_position_to_state
        self.session_id = session_id
        
        # Realistic trading constraints
        self.min_holding_periods = min_holding_periods
        self.max_trades_per_episode = max_trades_per_episode
        self.slippage_ticks = slippage_ticks
        
        # Tracking variables
        self.holding_time = 0
        self.trades_this_episode = 0
        self.last_trade_time = None
        
        # Standard environment variables
        self.done = False
        self.current_index = 0
        self.current_price = None
        self.current_position = 0
        self.last_position = 0
        
        self.entry_time = None
        self.entry_id = None
        self.entry_price = None
        
        self.exit_time = None
        self.exit_id = None
        self.exit_price = None
        
        self._last_closed_entry_price = None
        self._last_closed_exit_price = None
        
        self.total_reward = 0
        self.total_net_profit = 0
        self.orders = []
        self.executed_trades = []
        
        # Anti-exploitation tracking
        self.states_traded = set()  # Track which states have been traded
        self.last_trade_index = -10  # Ensure minimum gap between trades
        self.trades_at_current_state = 0  # Prevent multiple trades at same state
        
        # Setup trading logger
        if enable_trading_logger:
            from trading_logger import TradingLogger
            log_dir = f"logs/trading/session_{session_id}" if session_id else "logs/trading"
            self.trading_logger = TradingLogger(log_dir=log_dir)
        else:
            self.trading_logger = None
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # buy, hold, sell
        
        # Get the shape from the first state
        if len(self.states) > 0:
            first_state = self.states[0]
            if hasattr(first_state, 'data'):
                if hasattr(first_state.data, 'shape'):
                    state_shape = first_state.data.shape
                else:
                    state_shape = (len(first_state.data),) if hasattr(first_state.data, '__len__') else (1,)
            elif hasattr(first_state, 'shape'):
                state_shape = first_state.shape
            else:
                state_shape = (1,)  # Default shape
            
            if self.add_current_position_to_state:
                state_shape = (state_shape[0] + 1,)
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=state_shape, dtype=np.float32
            )
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed based on constraints"""
        # Check if already traded at this state (anti-exploitation)
        if self.current_index in self.states_traded:
            trading_logger.debug(f"Already traded at state index {self.current_index}")
            return False
        
        # Check minimum gap between trades (anti-exploitation)
        if self.current_index - self.last_trade_index < 5:
            trading_logger.debug(f"Too soon since last trade: {self.current_index - self.last_trade_index} steps")
            return False
        
        # Check trade limit
        if self.trades_this_episode >= self.max_trades_per_episode:
            trading_logger.debug(f"Trade limit reached: {self.trades_this_episode}/{self.max_trades_per_episode}")
            return False
        
        # Check holding period
        if self.current_position != 0 and self.holding_time < self.min_holding_periods:
            trading_logger.debug(f"Minimum holding period not met: {self.holding_time}/{self.min_holding_periods}")
            return False
        
        return True
    
    def _apply_slippage(self, price: float, direction: int) -> float:
        """Apply realistic slippage to order fills"""
        # Random slippage between 0 and slippage_ticks
        slippage = np.random.randint(0, self.slippage_ticks + 1) * self.tick_size
        
        if direction == 1:  # Buying
            return price + slippage  # Pay more
        else:  # Selling
            return price - slippage  # Receive less
    
    def step(self, action):
        """Execute one time step with realistic constraints"""
        state = self.states[self.current_index]
        self.current_price = state.price
        
        # Update holding time
        if self.current_position != 0:
            self.holding_time += 1
        
        # Map action to function
        if action == 0:
            self.buy(state)
        elif action == 2:
            self.sell(state)
        # action == 1 is hold (do nothing)
        
        # Get reward
        reward = self.get_reward(state)
        self.total_reward += reward
        
        # Move to next state
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True
        
        # Get next observation
        if not self.done:
            next_state = self.states[self.current_index]
            obs = self._get_observation(next_state)
        else:
            obs = np.zeros(self.observation_space.shape)
        
        return obs, reward, self.done, {}
    
    def buy(self, state: TimeSeriesState):
        """Execute buy order with realistic constraints"""
        if not self._can_trade():
            return
        
        # Check if order fills (probabilistic)
        if np.random.random() > self.fill_probability:
            trading_logger.debug(f"Buy order not filled at {state.ts}")
            return
        
        if self.current_position == -1:
            # Closing short position
            self.exit_price = self._apply_slippage(state.price, 1)
            self.exit_time = state.ts
            self._last_closed_entry_price = self.entry_price
            self._last_closed_exit_price = self.exit_price
            
            # Log trade exit
            if self.trading_logger:
                self.trading_logger.log_trade_exit(
                    timestamp=state.ts,
                    position_type='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.exit_price,
                    session_id=self.session_id,
                    state_info={'current_price': state.price}
                )
            
            self.current_position = 0
            self.holding_time = 0
            self.trades_this_episode += 1
            
            # Track successful trade (anti-exploitation)
            self.states_traded.add(self.current_index)
            self.last_trade_index = self.current_index
            
        elif self.current_position == 0:
            # Opening long position
            self.last_position = self.current_position
            self.current_position = 1
            self.entry_price = self._apply_slippage(state.price, 1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.holding_time = 0
            
            # Track successful trade (anti-exploitation)
            self.states_traded.add(self.current_index)
            self.last_trade_index = self.current_index
            
            # Log trade entry
            if self.trading_logger:
                self.trading_logger.log_trade_entry(
                    timestamp=state.ts,
                    position_type='LONG',
                    entry_price=self.entry_price,
                    target_price=state.price,
                    session_id=self.session_id,
                    state_info={'current_price': state.price}
                )
    
    def sell(self, state: TimeSeriesState):
        """Execute sell order with realistic constraints"""
        if not self._can_trade():
            return
        
        # Check if order fills
        if np.random.random() > self.fill_probability:
            trading_logger.debug(f"Sell order not filled at {state.ts}")
            return
        
        if self.current_position == 1:
            # Closing long position
            self.exit_price = self._apply_slippage(state.price, -1)
            self.exit_time = state.ts
            self._last_closed_entry_price = self.entry_price
            self._last_closed_exit_price = self.exit_price
            
            # Log trade exit
            if self.trading_logger:
                self.trading_logger.log_trade_exit(
                    timestamp=state.ts,
                    position_type='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.exit_price,
                    session_id=self.session_id,
                    state_info={'current_price': state.price}
                )
            
            self.current_position = 0
            self.holding_time = 0
            self.trades_this_episode += 1
            
            # Track successful trade (anti-exploitation)
            self.states_traded.add(self.current_index)
            self.last_trade_index = self.current_index
            
        elif self.current_position == 0:
            # Opening short position
            self.last_position = self.current_position
            self.current_position = -1
            self.entry_price = self._apply_slippage(state.price, -1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.holding_time = 0
            
            # Track successful trade (anti-exploitation)
            self.states_traded.add(self.current_index)
            self.last_trade_index = self.current_index
            
            # Log trade entry
            if self.trading_logger:
                self.trading_logger.log_trade_entry(
                    timestamp=state.ts,
                    position_type='SHORT',
                    entry_price=self.entry_price,
                    target_price=state.price,
                    session_id=self.session_id,
                    state_info={'current_price': state.price}
                )
    
    def get_reward(self, state: TimeSeriesState) -> float:
        """Calculate reward using realistic reward function"""
        # No reward for opening positions
        if self.last_position == 0 and self.current_position != 0:
            return 0.0
        
        # Calculate reward for closed positions
        if self.current_position == 0 and self.last_position != 0:
            if self.last_position == 1:  # Closed long
                return calculate_reward(
                    timestamp=state.ts,
                    action='EXIT',
                    position_before=self.last_position,
                    position_after=self.current_position,
                    entry_price=self._last_closed_entry_price,
                    exit_price=self._last_closed_exit_price,
                    position_type='long',
                    value_per_tick=self.value_per_tick,
                    execution_cost=2 * self.execution_cost_per_order,
                    session_id=self.session_id,
                    tick_size=self.tick_size
                )
            else:  # Closed short
                return calculate_reward(
                    timestamp=state.ts,
                    action='EXIT',
                    position_before=self.last_position,
                    position_after=self.current_position,
                    entry_price=self._last_closed_entry_price,
                    exit_price=self._last_closed_exit_price,
                    position_type='short',
                    value_per_tick=self.value_per_tick,
                    execution_cost=2 * self.execution_cost_per_order,
                    session_id=self.session_id,
                    tick_size=self.tick_size
                )
        
        # Small reward for holding (encourages patience)
        if self.current_position != 0 and self.entry_price is not None:
            position_type = 'long' if self.current_position == 1 else 'short'
            return calculate_reward(
                timestamp=state.ts,
                action='HOLD',
                position_before=self.current_position,
                position_after=self.current_position,
                entry_price=self.entry_price,
                exit_price=state.price,
                position_type=position_type,
                value_per_tick=self.value_per_tick,
                execution_cost=0,
                session_id=self.session_id,
                tick_size=self.tick_size
            )
        
        return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        # Handle seed if provided (for compatibility)
        if seed is not None:
            np.random.seed(seed)
        self.current_index = 0
        self.done = False
        self.current_position = 0
        self.last_position = 0
        self.holding_time = 0
        self.trades_this_episode = 0
        self.total_reward = 0
        self.total_net_profit = 0
        
        self.entry_price = None
        self.exit_price = None
        self._last_closed_entry_price = None
        self._last_closed_exit_price = None
        
        # Reset anti-exploitation tracking
        self.states_traded.clear()
        self.last_trade_index = -10
        self.trades_at_current_state = 0
        
        if len(self.states) > 0:
            return self._get_observation(self.states[0])
        else:
            return np.zeros(self.observation_space.shape)
    
    def _get_observation(self, state: TimeSeriesState):
        """Get observation with optional position information"""
        if hasattr(state, 'flatten'):
            obs = state.flatten()
        elif hasattr(state, 'data'):
            data = state.data
            # If it's a DataFrame, exclude non-numeric columns
            if hasattr(data, 'select_dtypes'):
                numeric_data = data.select_dtypes(include=[np.number])
                obs = numeric_data.values.flatten()
            elif hasattr(data, 'values'):
                obs = data.values.flatten()
            else:
                obs = np.array(data).flatten()
        else:
            # If state is already an array
            obs = np.array(state).flatten()
        
        # Ensure all values are numeric
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.add_current_position_to_state:
            obs = np.append(obs, self.current_position)
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_index}, Position: {self.current_position}, "
                  f"Trades: {self.trades_this_episode}, Holding Time: {self.holding_time}")


# Instructions for implementation:
print("""
To implement this realistic trading environment:

1. Save this file as gym_futures/envs/futures_env_realistic.py

2. Update your training_engine.py to use RealisticFuturesEnv:
   - Change: from gym_futures.envs import FuturesEnv
   - To: from gym_futures.envs.futures_env_realistic import RealisticFuturesEnv

3. Update environment creation with realistic parameters:
   env = RealisticFuturesEnv(
       states=states,
       value_per_tick=5.0,  # NQ value per tick
       tick_size=0.25,      # NQ tick size
       execution_cost_per_order=5.0,  # $5 per side
       min_holding_periods=10,  # Hold for at least 10 time steps
       max_trades_per_episode=5,  # Maximum 5 trades per episode
       slippage_ticks=2,  # 0-2 tick slippage
       session_id=session_id
   )

4. Restart training with these realistic constraints

Expected improvements:
- Fewer but more meaningful trades
- Realistic profit/loss patterns
- Better preparation for live trading
- No instant trading exploits
""")