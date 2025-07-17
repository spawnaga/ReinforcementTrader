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
    """Calculate the reward for a trading action with realistic costs
    
    Note: position_before is kept for API compatibility but not used in calculations
    """
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
        
        # Episode tracking for curriculum learning
        self.episode_number = 0
        
        # Setup trading logger
        if enable_trading_logger:
            from trading_logger import TradingLogger
            # Use simple logs directory - don't create subdirectories for each session
            self.trading_logger = TradingLogger(log_dir="logs")
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
        else:
            # Default observation space when no states are available
            default_shape = (60 + 1,) if self.add_current_position_to_state else (60,)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=default_shape, dtype=np.float32
            )
    
    def _can_trade(self) -> bool:
        """Check if trading is allowed based on constraints"""
        # Debug for episode 52
        if hasattr(self, 'episode_number') and self.episode_number == 52 and self.current_index < 5:
            trading_logger.error(
                f"Episode 52 _can_trade CHECK at index {self.current_index}: "
                f"states_traded={self.current_index in self.states_traded}, "
                f"gap_check={self.current_index - self.last_trade_index}, "
                f"trade_limit={self.trades_this_episode}/{self.max_trades_per_episode}, "
                f"holding_check={self.holding_time}/{self.min_holding_periods} if position"
            )
        
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
        
        # Store position before action for reward calculation
        self.last_position = self.current_position
        
        # Map action to function
        if action == 0:
            self.buy(state)
        elif action == 2:
            self.sell(state)
        # action == 1 is hold (do nothing)
        
        # Get reward
        reward = self.get_reward(state)
        
        # CRITICAL DEBUG: Check if reward contains suspicious value
        if abs(reward) > 50:  # Individual step rewards should never be this large
            trading_logger.error(
                f"SUSPICIOUS REWARD VALUE: Episode {self.episode_number}, Step {self.current_index}: "
                f"get_reward() returned {reward:.2f} | price={state.price:.2f} | "
                f"entry_price={self.entry_price} | total_net_profit={self.total_net_profit}"
            )
            # Check if reward is accidentally a price value
            if abs(reward - state.price) < 1 or abs(reward - state.price*4) < 1:
                trading_logger.error(
                    f"*** BUG FOUND: Reward {reward:.2f} looks like price data! ***"
                )
        
        self.total_reward += reward
        
        # Debug first step rewards in problematic episodes
        if self.episode_number >= 50 and self.current_index < 3:
            trading_logger.info(
                f"Episode {self.episode_number}, Step {self.current_index}: "
                f"action={action}, reward={reward:.2f}, total_reward={self.total_reward:.2f}, "
                f"position={self.current_position}, last_position={self.last_position}, "
                f"trades={self.trades_this_episode}"
            )
        
        # Debug massive rewards
        if abs(self.total_reward) > 10000 and self.trades_this_episode == 0:
            trading_logger.error(
                f"MASSIVE TOTAL REWARD WITHOUT TRADES: ${self.total_reward:.2f} at step {self.current_index}, "
                f"step_reward: ${reward:.2f}, trades: {self.trades_this_episode}"
            )
        
        # Move to next state
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True
            # Debug log when episode ends
            if self.episode_number >= 50 and self.trades_this_episode == 0:
                trading_logger.error(
                    f"Episode {self.episode_number} ENDING: total_reward={self.total_reward:.2f}, "
                    f"last step reward={reward:.2f}, trades={self.trades_this_episode}"
                )
        
        # Get next observation
        if not self.done:
            next_state = self.states[self.current_index]
            obs = self._get_observation(next_state)
        else:
            obs = np.zeros(self.observation_space.shape)
            
        # Debug: Check if the observation contains large values
        if self.episode_number >= 50 and abs(self.total_reward) > 10000:
            if not self.done and np.max(np.abs(obs)) > 10000:
                trading_logger.error(
                    f"Episode {self.episode_number}, Step {self.current_index}: "
                    f"Observation contains large value! Max abs value: {np.max(np.abs(obs)):.2f}"
                )
        
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
        """Calculate reward using realistic reward function with shaping and exploration bonus"""
        # EXPLORATION BONUS: Small reward for opening positions in early episodes
        # This encourages the agent to explore trading rather than staying flat
        if self.last_position == 0 and self.current_position != 0:
            if hasattr(self, 'episode_number') and self.episode_number < 100:
                # Larger exploration bonus in early episodes, decreasing over time
                exploration_bonus = 0.5 * (1.0 - self.episode_number / 100.0)
                return exploration_bonus
            return 0.0
        
        # Calculate reward for closed positions
        if self.current_position == 0 and self.last_position != 0:
            base_reward = 0.0
            if self.last_position == 1:  # Closed long
                base_reward = calculate_reward(
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
                base_reward = calculate_reward(
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
            
            # CURRICULUM-BASED REWARD SHAPING
            if self.episode_number < 50:
                # EASY STAGE: Small bonuses for profitable trades, reduced penalties for losses
                if base_reward > 0:
                    # Add small fixed bonus for profitable trades (not percentage based!)
                    # This encourages trading without distorting the reward signal
                    fixed_bonus = 5.0  # Small $5 bonus for any profitable trade
                    total_reward = base_reward + fixed_bonus
                    
                    if self.trading_logger:
                        self.trading_logger.log_reward_calculation(
                            "CURRICULUM EASY: PROFITABLE TRADE",
                            timestamp=state.ts,
                            details={
                                'episode': self.episode_number,
                                'base_reward': base_reward,
                                'fixed_bonus': fixed_bonus,
                                'total_reward': total_reward
                            }
                        )
                    return total_reward
                else:
                    # Reduce losses by 10% to encourage exploration (not 30%!)
                    loss_reduction = abs(base_reward) * 0.1
                    reduced_loss = base_reward + loss_reduction
                    
                    if self.trading_logger:
                        self.trading_logger.log_reward_calculation(
                            "CURRICULUM EASY: LOSS REDUCTION",
                            timestamp=state.ts,
                            details={
                                'episode': self.episode_number,
                                'base_loss': base_reward,
                                'reduction': loss_reduction,
                                'final_loss': reduced_loss
                            }
                        )
                    return max(reduced_loss, -200.0)  # Cap losses at -$200 in easy mode
                    
            elif self.episode_number < 150:
                # MEDIUM STAGE: Smaller bonuses
                if base_reward > 0:
                    # Small fixed bonus, no percentage multiplier
                    fixed_bonus = 3.0  # $3 bonus for profitable trades
                    total_reward = base_reward + fixed_bonus
                    
                    if self.trading_logger:
                        self.trading_logger.log_reward_calculation(
                            "CURRICULUM MEDIUM: PROFITABLE TRADE",
                            timestamp=state.ts,
                            details={
                                'episode': self.episode_number,
                                'base_reward': base_reward,
                                'fixed_bonus': fixed_bonus,
                                'total_reward': total_reward
                            }
                        )
                    return total_reward
                else:
                    # Small loss reduction
                    loss_reduction = abs(base_reward) * 0.15
                    return max(base_reward + loss_reduction, -350.0)
                    
            else:
                # HARD STAGE: No bonuses (realistic)
                if base_reward > 0:
                    # No bonus in hard mode - pure profit/loss
                    total_reward = base_reward
                    
                    if self.trading_logger:
                        self.trading_logger.log_reward_calculation(
                            "CURRICULUM HARD: PROFITABLE TRADE",
                            timestamp=state.ts,
                            details={
                                'episode': self.episode_number,
                                'base_reward': base_reward,
                                'total_reward': total_reward
                            }
                        )
                    return total_reward
                else:
                    # Full losses, realistic trading
                    return max(base_reward, -500.0)  # Cap losses at -$500 per trade
        
        # Small reward for holding (encourages patience) but penalize holding too long
        if self.current_position != 0 and self.entry_price is not None:
            position_type = 'long' if self.current_position == 1 else 'short'
            hold_reward = calculate_reward(
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
            
            # Add penalty for holding too long to encourage closing positions
            if self.holding_time > 100:  # After 100 steps
                holding_penalty = -0.5 * (self.holding_time - 100) / 100  # Gradually increase penalty
                hold_reward += holding_penalty
                
            # Clip holding rewards to prevent exploitation
            return np.clip(hold_reward, -100, 100)
        
        # If we're flat and have made some trades, no penalty
        if self.current_position == 0 and self.trades_this_episode > 0:
            return 0.0
            
        # Small penalty for not trading to encourage action, but only if truly not trading
        if self.trades_this_episode == 0 and self.current_index > 50:
            # Curriculum-based penalty - smaller in early episodes
            if self.episode_number < 50:
                penalty = -0.05  # Very small penalty in easy mode
            elif self.episode_number < 150:
                penalty = -0.075  # Small penalty in medium mode
            else:
                penalty = -0.1  # Normal penalty in hard mode
                
            # Debug logging for no-trade penalties
            if self.trading_logger and self.current_index == 51:
                self.trading_logger.info(
                    f"NO TRADE PENALTY: Episode {self.episode_number}, "
                    f"applying {penalty} penalty per step for not trading"
                )
            
            return penalty
        
        return 0.0
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode"""
        # Handle seed if provided (for compatibility)
        if seed is not None:
            np.random.seed(seed)
            
        # Debug log total_reward before reset
        if hasattr(self, 'episode_number') and self.episode_number >= 50:
            trading_logger.info(
                f"RESET Episode {self.episode_number}: total_reward before reset = {getattr(self, 'total_reward', 'NOT SET')}"
            )
            
        # CURRICULUM LEARNING: Adjust episode parameters based on episode number
        self._apply_curriculum_learning()
        
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
        
        # CRITICAL FIX: Reset anti-exploitation tracking for new episode
        self.states_traded.clear()  # Clear the set of traded indices
        self.last_trade_index = -10  # Reset to allow immediate trading
        self.trades_at_current_state = 0  # Reset trades at current state
        
        # Increment episode number for next episode
        self.episode_number += 1
        
        # Debug log reset state for episodes with issues
        if self.episode_number >= 50:
            trading_logger.info(
                f"Episode {self.episode_number} RESET COMPLETE: "
                f"trades_this_episode={self.trades_this_episode}, "
                f"last_trade_index={self.last_trade_index}, "
                f"states_traded={len(self.states_traded)}, "
                f"total_reward={self.total_reward}, "
                f"max_trades={self.max_trades_per_episode}, "
                f"min_holding={self.min_holding_periods}"
            )
        
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
    
    def _apply_curriculum_learning(self):
        """Apply curriculum learning by adjusting difficulty based on episode number"""
        # CURRICULUM STAGES:
        # Stage 1 (Episodes 0-50): Easy - trending markets, more trades allowed, smaller penalties
        # Stage 2 (Episodes 51-150): Medium - normal market conditions
        # Stage 3 (Episodes 151+): Hard - more volatile markets, stricter constraints
        
        if self.episode_number < 50:
            # EASY STAGE: Help agent learn basic trading
            self.max_trades_per_episode = 10  # Allow more trades to explore
            self.min_holding_periods = 5  # Shorter holding requirement
            self.execution_cost_per_order = 2.5  # Lower costs
            self.slippage_ticks = 1  # Less slippage
            
            # Select trending market data (easier to predict)
            # In real implementation, this would filter for trending periods
            # For now, we use the same data but with easier parameters
            
            if self.trading_logger:
                self.trading_logger.info(f"CURRICULUM: Easy stage (Episode {self.episode_number})")
                
        elif self.episode_number < 150:
            # MEDIUM STAGE: Normal trading conditions
            self.max_trades_per_episode = 7  # Moderate trade limit
            self.min_holding_periods = 8  # Medium holding period
            self.execution_cost_per_order = 5.0  # Normal costs
            self.slippage_ticks = 2  # Normal slippage
            
            if self.trading_logger:
                self.trading_logger.info(f"CURRICULUM: Medium stage (Episode {self.episode_number})")
                
        else:
            # HARD STAGE: Challenging conditions
            self.max_trades_per_episode = 5  # Strict trade limit
            self.min_holding_periods = 10  # Long holding period
            self.execution_cost_per_order = 7.5  # Higher costs
            self.slippage_ticks = 3  # More slippage
            
            # Could also:
            # - Select more volatile market periods
            # - Reduce episode length
            # - Add market impact modeling
            
            if self.trading_logger:
                self.trading_logger.info(f"CURRICULUM: Hard stage (Episode {self.episode_number})")
        
        # Gradually reduce episode length to make it harder
        # Start with full data, then use less as training progresses
        if self.episode_number < 50:
            self.limit = min(len(self.states), 300)  # Shorter episodes initially
        elif self.episode_number < 150:
            self.limit = min(len(self.states), 200)  # Standard episode length
        else:
            self.limit = min(len(self.states), 150)  # Shorter, harder episodes
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_index}, Position: {self.current_position}, "
                  f"Trades: {self.trades_this_episode}, Holding Time: {self.holding_time}")


# This file implements the RealisticFuturesEnv with proper constraints