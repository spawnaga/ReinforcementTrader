import datetime
import logging
from pathlib import Path
from typing import List, Tuple, Sequence, Optional
from uuid import uuid4

import gym
import numpy as np
from gym import spaces

from .utils import round_to_nearest_increment, TimeSeriesState

logger = logging.getLogger(__name__)

# Setup detailed trading log file
import os
os.makedirs('logs', exist_ok=True)  # Create logs directory if it doesn't exist
trading_logger = logging.getLogger('trading')
trading_handler = logging.FileHandler('logs/trading.log')
trading_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
trading_logger.addHandler(trading_handler)
trading_logger.setLevel(logging.DEBUG)


def calculate_reward(timestamp, action, position_before, position_after, entry_price, exit_price, position_type,
                     value_per_tick, execution_cost, session_id, tick_size):
    """
    Calculate the reward for a trading action.

    Parameters:
    - timestamp: The timestamp of the action
    - action: 'EXIT' for closing a position, 'HOLD' for holding
    - position_before: Position before the action (1 long, -1 short, 0 none)
    - position_after: Position after the action
    - entry_price: Entry price of the position
    - exit_price: Exit price for exit, current price for hold
    - position_type: 'long' or 'short'
    - value_per_tick: Value per tick movement
    - execution_cost: Execution cost (full for exit, 0 for hold typically)
    - session_id: Session ID
    - tick_size: The tick size for the instrument

    Returns:
    - reward: The calculated reward
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
        net_profit = profit_ticks * value_per_tick - execution_cost
        # Log the reward calculation if needed
        trading_logger.debug(
            f"REWARD CALC: timestamp={timestamp}, action={action}, position_type={position_type}, entry={entry_price}, exit={exit_price}, profit_ticks={profit_ticks}, net_profit={net_profit}"
        )
        return net_profit
    elif action == 'HOLD' and position_after != 0:
        # Unrealized profit with no execution cost
        if position_type == 'long':
            unrealized_ticks = (exit_price - entry_price) / tick_size
        elif position_type == 'short':
            unrealized_ticks = (entry_price - exit_price) / tick_size
        else:
            return 0.0
        unrealized = unrealized_ticks * value_per_tick
        # Perhaps scale it down for hold rewards
        return unrealized * 0.1  # Adjust factor as needed
    else:
        return 0.0


class FuturesEnv(gym.Env):
    """
    A gym for training futures trading RL environments.

    The futures market is different than a typical stock trading environment, in that
    contracts move in fixed increments, and each increment (tick) is worth a variable
    amount depending on the contract traded.

    This environment is designed for a single contract - for a single security type.
    Scaling the agent trained is only a matter of scaling up the order size (within reasonable)
    limits.

    Accompanying this environment is the concept of a TimeSeriesState, which is a variable
    2-d length window of all the features representing the state, or something that has
    additional dimensions (e.g a stack of scalograms). See `TimeSeriesState` for more
    details and examples

    This environment accepts 3 different actions at any time state:
      0 - a buy
      1 - a hold (no action)
      2 - a sell

    The environment does not allow for more than one contract to be traded at a time.
    If a buy action is submitted with an existing long trade, the action defaults to
    no action.

    You can add the current position to the next state sent to the agent by setting
    `add_current_trade_information_to_state` to True.

    This environment can also simulate the probabilistic dynamics of actual market trading
    through the `fill_probability` and long/short probability vectors. Occasionally,
    an order will not fill, or will fill at a price that differs from the intended
    price. See `generate_random_fill_differential` for usage details. If deterministic
    behavior is desired, do not supply these arguments


    The standard reward function is the net_profit
    where net_profit is equal to
         ((entry_price - exit_price) / tick_size) * value_per_tick)
        - execution_cost_per_order * 2

    It's likely that you will want to use a more complex reward function.
    In this case, subclass this environment and overwrite `get_reward()`


    Attributes
    ----------
    states: Sequence[TimeSeriesState]
      a sequence of `TimeSeriesState` objects
    value_per_tick: float
      the value per 1 tick movement. E.g. 1 tick movement for ES is 12.50, for CL is 10
    tick_size: float
      the minimum movement in the price, or tick size. E.g. ES = 0.25, CL = 0.01
    fill_probability: float
      the probability of filling a submitted order. Defaults to 1
    long_values: List[float]
      a list of values that represent possible differentials from the
      intended fill price for buy orders
    long_probabilities: List[float]
      the probability distribution for the possible differentials
      specified in long_values. the length must equal the length of long_values and the sum
      of the probabilities must equal 1
    short_values: List[float]
      a list of values that represent possible differentials from the
      intended fill price for sell_orders
    short_probabilities: List[float]
      the probability distribution for the possible differentials
      specified in short_values. the length must equal the length of
      short_values and the sum of the probabilities must equal 1
    execution_cost_per_trade: float
      the cost of executing 1 buy or sell order. Include the cost
      per 1 side of the trade, the calculation for net profit
      accounts for 2 orders
    add_current_position_to_state: bool
      adds the current position to the next state. Default: False
    log_dir: str
      a str or Path representing the directory to
      render the results of the epoch. see `render` will generate
      output metrics to tensorflow
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, states: Sequence[TimeSeriesState], value_per_tick: float,
                 tick_size: float, fill_probability: float = 1., long_values: List[float] = None,
                 long_probabilities: List[float] = None, short_values: List[float] = None,
                 short_probabilities: List[float] = None, execution_cost_per_order=0.,
                 add_current_position_to_state: bool = False,
                 log_dir: str = f"logs/futures_env/{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')}",
                 enable_trading_logger: bool = True, session_id: Optional[int] = None):

        self.states = states
        self.limit = len(self.states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.long_values = long_values
        self.long_probabilities = long_probabilities
        self.short_values = short_values
        self.short_probabilities = short_probabilities
        self.can_generate_random_fills = all(
            [self.long_values, self.long_probabilities, self.short_values, self.short_probabilities])
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.add_current_position_to_state = add_current_position_to_state
        self.log_dir = log_dir
        self.done = False
        self.current_index = 0
        self.session_id = session_id

        self.current_price = None
        # attributes to maintain the current position
        self.current_position = 0
        self.last_position = 0

        self.entry_time = None
        self.entry_id = None
        self.entry_price = None

        self.exit_time = None
        self.exit_id = None
        self.exit_price = None

        # Store last closed trade prices for reward calculation
        self._last_closed_entry_price = None
        self._last_closed_exit_price = None

        # episode attributes
        self.total_reward = 0
        self.total_net_profit = 0
        self.orders = []
        self.trades = []
        self.episode = 0
        self.feature_data = []

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=buy, 1=hold, 2=sell

        # Observation space will be defined based on the state data
        if self.states:
            sample_state = self.states[0]
            if hasattr(sample_state, 'data'):
                if hasattr(sample_state.data, 'shape'):
                    obs_shape = sample_state.data.shape
                else:
                    obs_shape = (len(sample_state.data),) if hasattr(sample_state.data, '__len__') else (1,)
            else:
                obs_shape = (1,)

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize trading logger
        self.enable_trading_logger = enable_trading_logger
        if self.enable_trading_logger:
            try:
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
                from trading_logger import TradingLogger
                self.trading_logger = TradingLogger(log_dir=self.log_dir)
                logger.info(f"Trading logger initialized for session: {self.trading_logger.session_timestamp}")
            except Exception as e:
                logger.warning(f"Could not initialize trading logger: {e}")
                self.trading_logger = None
        else:
            self.trading_logger = None

    def buy(self, state: TimeSeriesState):
        """Creates a buy order"""
        if self.current_position == 1:
            # does not perform a buy order
            pass
        elif self.current_position == -1:
            self.last_position = self.current_position
            self.current_position = 0

            self.exit_price = self.generate_random_fill_differential(state.price, 1)
            self.exit_time = state.ts
            self.exit_id = str(uuid4())
            self.orders.append([self.exit_id, str(state.ts), self.exit_price, 1, state])

            # Log trade exit to file
            trading_logger.debug(
                f"TRADE EXIT: timestamp={state.ts}, position_type=SHORT, entry_price={self.entry_price}, exit_price={self.exit_price}, state_info={{'current_price': {state.price}, 'position_before': -1, 'position_after': 0}}")

            if self.trading_logger:
                # Calculate profit/loss for the trade
                profit_loss = None
                if self.entry_price and self.exit_price:
                    profit_ticks = (self.entry_price - self.exit_price) / self.tick_size
                    profit_loss = profit_ticks * self.value_per_tick - (2 * self.execution_cost_per_order)
                
                self.trading_logger.log_trade_exit(
                    timestamp=state.ts,
                    position_type='SHORT',
                    entry_price=self.entry_price,
                    exit_price=self.exit_price,
                    profit_loss=profit_loss,
                    session_id=self.session_id,
                    state_info={'current_price': state.price, 'position_before': -1, 'position_after': 0}
                )
                self.trading_logger.log_position_change(
                    timestamp=state.ts,
                    old_position=-1,
                    new_position=0,
                    reason='Buy order closing short position',
                    price=self.exit_price
                )

            # Store prices for reward calculation before resetting
            self._last_closed_entry_price = self.entry_price
            self._last_closed_exit_price = self.exit_price

            # Reset prices after closing position
            self.entry_price = None
            self.exit_price = None

        elif self.current_position == 0:
            self.last_position = self.current_position
            self.current_position = 1
            self.entry_price = self.generate_random_fill_differential(state.price, 1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.orders.append([self.entry_id, str(state.ts), self.entry_price, 1, state])

            # Log trade entry to file
            trading_logger.debug(
                f"TRADE ENTRY: timestamp={state.ts}, position_type=LONG, entry_price={self.entry_price}, target_price={state.price}, state_info={{'current_price': {state.price}, 'position_before': 0, 'position_after': 1}}")

            if self.trading_logger:
                self.trading_logger.log_trade_entry(
                    timestamp=state.ts,
                    position_type='LONG',
                    entry_price=self.entry_price,
                    target_price=state.price,
                    session_id=self.session_id,
                    state_info={'current_price': state.price, 'position_before': 0, 'position_after': 1}
                )
                self.trading_logger.log_position_change(
                    timestamp=state.ts,
                    old_position=0,
                    new_position=1,
                    reason='Buy order opening long position',
                    price=self.entry_price
                )

    def sell(self, state: TimeSeriesState):
        """Creates a sell order"""
        if self.current_position == -1:
            # already short, so does nothing
            pass
        elif self.current_position == 1:
            self.last_position = self.current_position
            self.current_position = 0

            self.exit_price = self.generate_random_fill_differential(state.price, -1)
            self.exit_time = state.ts
            self.exit_id = str(uuid4())
            self.orders.append([self.exit_id, str(state.ts), self.exit_price, -1, state])

            # Log trade exit to file
            trading_logger.debug(
                f"TRADE EXIT: timestamp={state.ts}, position_type=LONG, entry_price={self.entry_price}, exit_price={self.exit_price}, state_info={{'current_price': {state.price}, 'position_before': 1, 'position_after': 0}}")

            if self.trading_logger:
                # Calculate profit/loss for the trade
                profit_loss = None
                if self.entry_price and self.exit_price:
                    profit_ticks = (self.exit_price - self.entry_price) / self.tick_size
                    profit_loss = profit_ticks * self.value_per_tick - (2 * self.execution_cost_per_order)
                
                self.trading_logger.log_trade_exit(
                    timestamp=state.ts,
                    position_type='LONG',
                    entry_price=self.entry_price,
                    exit_price=self.exit_price,
                    profit_loss=profit_loss,
                    session_id=self.session_id,
                    state_info={'current_price': state.price, 'position_before': 1, 'position_after': 0}
                )
                self.trading_logger.log_position_change(
                    timestamp=state.ts,
                    old_position=1,
                    new_position=0,
                    reason='Sell order closing long position',
                    price=self.exit_price
                )

            # Store prices for reward calculation before resetting
            self._last_closed_entry_price = self.entry_price
            self._last_closed_exit_price = self.exit_price

            # Reset prices after closing position
            self.entry_price = None
            self.exit_price = None

        elif self.current_position == 0:
            self.last_position = self.current_position
            self.current_position = -1
            self.entry_price = self.generate_random_fill_differential(state.price, -1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.orders.append([self.entry_id, str(state.ts), self.entry_price, -1, state])

            # Log trade entry to file
            trading_logger.debug(
                f"TRADE ENTRY: timestamp={state.ts}, position_type=SHORT, entry_price={self.entry_price}, target_price={state.price}, state_info={{'current_price': {state.price}, 'position_before': 0, 'position_after': -1}}")

            if self.trading_logger:
                self.trading_logger.log_trade_entry(
                    timestamp=state.ts,
                    position_type='SHORT',
                    entry_price=self.entry_price,
                    target_price=state.price,
                    session_id=self.session_id,
                    state_info={'current_price': state.price, 'position_before': 0, 'position_after': -1}
                )
                self.trading_logger.log_position_change(
                    timestamp=state.ts,
                    old_position=0,
                    new_position=-1,
                    reason='Sell order opening short position',
                    price=self.entry_price
                )

    def get_reward(self, state: TimeSeriesState) -> float:
        """
        Reward of: 
          - current state if position is flat
          - (current price - avg_price) if position is long
          - (avg_price - current_price) if position is short
        times net P&L conversion

        Returns
        -------
        reward: float
          The reward for the current state
        """
        # If the position changed this step, calculate the trade P&L
        if self.current_position == 0 and self.last_position != 0:
            # Just closed a position - use the stored entry/exit prices
            if self._last_closed_entry_price is not None and self._last_closed_exit_price is not None:
                position_type = 'long' if self.last_position == 1 else 'short'
                reward = calculate_reward(
                    timestamp=state.ts,
                    action='EXIT',
                    position_before=self.last_position,
                    position_after=self.current_position,
                    entry_price=self._last_closed_entry_price,
                    exit_price=self._last_closed_exit_price,
                    position_type=position_type,
                    value_per_tick=self.value_per_tick,
                    execution_cost=2 * self.execution_cost_per_order,
                    session_id=self.session_id,
                    tick_size=self.tick_size
                )
                # Log the reward
                if self.trading_logger:
                    self.trading_logger.log_reward_calculation(
                        "TRADE EXIT REWARD",
                        timestamp=state.ts,
                        details={
                            'position_type': position_type,
                            'entry_price': self._last_closed_entry_price,
                            'exit_price': self._last_closed_exit_price,
                            'reward': reward,
                            'session_id': self.session_id
                        }
                    )
                return reward
            else:
                # Should not happen, but safeguard
                return 0.0
        
        # Holding a position - calculate unrealized P&L (small reward for good positions)
        elif self.current_position != 0 and self.entry_price is not None:
            position_type = 'long' if self.current_position == 1 else 'short'
            reward = calculate_reward(
                timestamp=state.ts,
                action='HOLD',
                position_before=self.current_position,
                position_after=self.current_position,
                entry_price=self.entry_price,
                exit_price=state.price,
                position_type=position_type,
                value_per_tick=self.value_per_tick,
                execution_cost=0,  # No execution cost for holding
                session_id=self.session_id,
                tick_size=self.tick_size
            )
            return reward
        
        # Flat and not trading - no reward
        else:
            return 0.0

    def generate_random_fill_differential(self, price: float, direction: int) -> float:
        """
        generates a random fill differential to simulate the dynamics of trading
        
        A buy order will fill at a higher price and sell order will fill at a lower price
        """
        if not self.can_generate_random_fills:
            return price
        if direction == 1:
            return round_to_nearest_increment(
                np.random.choice(self.long_values, p=self.long_probabilities) + price,
                self.tick_size
            )
        else:
            return round_to_nearest_increment(
                np.random.choice(self.short_values, p=self.short_probabilities) + price,
                self.tick_size
            )

    def step(self, action):
        """
        Parameters
        ----------
        action: int
          0 = buy, 1 = hold (do nothing), 2 = sell
        
        Returns
        -------
        state: np.array
          array of prices up to the current position
        reward: float
          current step reward
        done: bool
          True if episode is over, False otherwise
        """
        state = self.states[self.current_index]
        self.current_price = state.price

        # store position before action for reward calculation
        self.last_position = self.current_position

        # map action to function
        if action == 0:
            self.buy(state)
        elif action == 2:
            self.sell(state)
        # action == 1 is hold (do nothing)

        # calculate reward
        reward = self.get_reward(state)
        self.total_reward += reward

        # check if we're done
        self.current_index += 1
        if self.current_index >= self.limit:
            self.done = True

        # get next observation
        if not self.done:
            next_state = self.states[self.current_index]
            obs = self._get_observation(next_state)
        else:
            obs = None

        return obs, reward, self.done, {}

    def _get_observation(self, state: TimeSeriesState):
        """Get observation from state"""
        if hasattr(state, 'data'):
            if isinstance(state.data, np.ndarray):
                return state.data.flatten().astype(np.float32)
            elif hasattr(state.data, 'values'):
                return state.data.values.flatten().astype(np.float32)
            else:
                return np.array(state.data).flatten().astype(np.float32)
        else:
            # Return price as a single feature if no other data
            return np.array([state.price], dtype=np.float32)

    def reset(self):
        """Reset the environment to the initial state"""
        self.done = False
        self.current_index = 0
        self.current_price = None
        
        # Reset positions
        self.current_position = 0
        self.last_position = 0
        
        # Reset entry/exit tracking
        self.entry_time = None
        self.entry_id = None
        self.entry_price = None
        self.exit_time = None
        self.exit_id = None
        self.exit_price = None
        self._last_closed_entry_price = None
        self._last_closed_exit_price = None
        
        # Reset episode tracking
        self.total_reward = 0
        self.total_net_profit = 0
        self.orders = []
        self.trades = []
        self.episode += 1
        self.feature_data = []
        
        # Get initial observation
        if self.states:
            return self._get_observation(self.states[0])
        else:
            return None

    def render(self, mode='human'):
        """Render the environment (not implemented)"""
        pass
    
    def close(self):
        """Close the environment"""
        pass