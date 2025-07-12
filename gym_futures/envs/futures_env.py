import gym
import json
import datetime
import pandas as pd
import numpy as np
import math
from pathlib import Path
from typing import List, Tuple, Sequence, Union, Dict
from gym import error, spaces, utils
from gym.utils import seeding
from .utils import round_to_nearest_increment, TimeSeriesState, monotonicity 
from uuid import uuid4

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
                 log_dir: str = f"logs/futures_env/{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')}"):
 
        self.states = states
        self.limit = len(self.states)
        self.value_per_tick = value_per_tick
        self.tick_size = tick_size
        self.long_values = long_values
        self.long_probabilities  = long_probabilities
        self.short_values = short_values
        self.short_probabilities = short_probabilities
        self.can_generate_random_fills = all([self.long_values, self.long_probabilities, self.short_values, self.short_probabilities])
        self.fill_probability = fill_probability
        self.execution_cost_per_order = execution_cost_per_order
        self.add_current_position_to_state = add_current_position_to_state
        self.log_dir = log_dir
        self.done = False
        self.current_index = 0

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

        elif self.current_position == 0:
            self.last_position = self.current_position
            self.current_position = 1
            self.entry_price = self.generate_random_fill_differential(state.price, 1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.orders.append([self.entry_id, str(state.ts), self.entry_price, 1, state])

    def sell(self, state: TimeSeriesState):
        """generates a sell order"""
        if self.current_position == -1:
            # does not perform a sell
            pass

        elif self.current_position == 1:
            self.last_position = self.current_position
            self.current_position = 0
            self.exit_price = self.generate_random_fill_differential(state.price, -1)
            self.exit_time = state.ts
            self.exit_id = str(uuid4())
            self.orders.append([self.exit_id, str(state.ts), self.exit_price, -1, state])

        elif self.current_position == 0:
            self.last_position = self.current_position
            self.current_position = -1
            self.entry_price = self.generate_random_fill_differential(state.price, -1)
            self.entry_time = state.ts
            self.entry_id = str(uuid4())
            self.orders.append([self.entry_id, str(state.ts), self.entry_price,-1, state])

    def get_reward(self, state: TimeSeriesState) -> float:
        """
        This environments default reward function. Override this class and method for a custom reward function
        """
        net_profit = 0
        if any([all([self.current_position == 0, self.last_position == 0]),
            all([self.current_position == 1, self.last_position == 0]),
            all([self.current_position == -1, self.last_position == 0])]):
            # no reward for no action taken or holding a position only receive rewards for closing a trade
            return net_profit

        else:
            if all([self.current_position == 0, self.last_position == 1]):
                # closed a long
                dif =  round((self.exit_price - self.entry_price),2)
            elif all([self.current_position == 0, self.last_position == -1]):
                # closed a short
                dif = - round((self.exit_price - self.entry_price),2)
            else:
                return net_profit
                
            n_ticks = math.ceil(dif / self.tick_size)
            gross_profit = n_ticks * self.value_per_tick
            net_profit = gross_profit - (2*self.execution_cost_per_order)
        
        self.total_reward += net_profit
        return net_profit

    def generate_random_fill_differential(self, intended_price: float, side: int) -> float:
        """
        Generate a random fill price based on the intended price and side
        
        Args:
            intended_price: The intended fill price
            side: 1 for buy, -1 for sell
            
        Returns:
            Actual fill price
        """
        if not self.can_generate_random_fills:
            return intended_price
        
        if side == 1:  # Buy order
            differential = np.random.choice(self.long_values, p=self.long_probabilities)
        else:  # Sell order
            differential = np.random.choice(self.short_values, p=self.short_probabilities)
        
        fill_price = intended_price + (differential * self.tick_size)
        return round_to_nearest_increment(fill_price, self.tick_size)

    def step(self, action):
        """
        This mimics OpenAIs training cycle, where the agent produces an action, and the action is provided to the step function of the environment.
        The environment will return the expected (next_state, reward, done, info) tuple

            _s == s' (next state)
             s == s (the current state that the action is for)

        """
        _s, s = self._get_next_state()
        if self.done:
            return (None, None, self.done, None)

        current_state_price = s.price
        next_state_price = _s.price if _s else current_state_price

        if action == 0:
            # a buy action signal is received
            if self.current_position == 1:
                # a buy is recommended whilst in a long - no action
                reward = self.get_reward(s) 
                info = {"message": "hold - a buy was recommended while in an open long position"}
                return (_s, reward, self.done, info)
            if self.current_position == 0:
                # a buy is recommended by the agent whilst no position - creating a long
                # this fills with pr(fill) == self.fill_probability
                if np.random.choice(a=[0,1], 
                                    size=1, 
                                    p=[1-self.fill_probability, self.fill_probability])[0] == 1:
                    
                    self.buy(s)

                    reward = self.get_reward(s)

                    info = {
                        "message": f"timestamp: {str(self.entry_time)}, long trade attempted at: {current_state_price}, filled at: {self.entry_price}"
                    }
                    return (_s, reward, self.done, info)
                else:
                    reward = 0
                    info= {
                        "message": "a long was recommended, but was not filled given the current fill probability"
                    }
                    return(_s, reward, self.done, info)


            if self.current_position == -1:
                # a buy is recommended by the agent whilst in a sell. 
                # This closes a short

                self.buy(s)  

                reward = self.get_reward(s)

                net_profit = reward

                info = {
                    "message": f"timestamp: {str(s.ts)}, short closed from {self.entry_price} to {self.exit_price} - total profit: {net_profit}"
                }

                self._close_position(reward, net_profit)

                return (_s, reward, self.done, info)

        elif action == 1:
            # no action recommended
            reward = self.get_reward(s)
            info = {"message": "no action performed"} 
            return (_s, reward, self.done, info)
        
        
        elif action == 2:
            # a sell signal is received
            if self.current_position == 1:
                # a sell is recommended by the agent whilst in a buy. 
                # This closes a long

                self.sell(s)  

                reward = self.get_reward(s)

                net_profit = reward

                info = {
                    "message": f"timestamp: {str(s.ts)}, long closed from {self.entry_price} to {self.exit_price} - total profit: {net_profit}"
                }

                self._close_position(reward, net_profit)

                return (_s, reward, self.done, info)

            if self.current_position == 0:
                # a sell is recommended by the agent whilst no position - creating a short
                # this fills with pr(fill) == self.fill_probability
                if np.random.choice(a=[0,1], 
                                    size=1, 
                                    p=[1-self.fill_probability, self.fill_probability])[0] == 1:
                    
                    self.sell(s)

                    reward = self.get_reward(s)

                    info = {
                        "message": f"timestamp: {str(self.entry_time)}, short trade attempted at: {current_state_price}, filled at: {self.entry_price}"
                    }
                    return (_s, reward, self.done, info)       
                
                else:
                    reward = 0
                    info = {
                        "message": "a short was recommended, but was not filled given the current fill probability"
                    }
                    return(_s, reward, self.done, info)

            if self.current_position == -1:
                # a sell is recommended whilst in a short - no action
                reward = self.get_reward(s) 
                info = {"message": "hold - a sell was recommended while in an open short position"}
                return (_s, reward, self.done, info)

    def _get_next_state(self) -> Tuple[TimeSeriesState, TimeSeriesState]:
        """Get the next state and current state"""
        if self.current_index >= len(self.states) - 1:
            self.done = True
            return None, self.states[self.current_index] if self.current_index < len(self.states) else None
        
        current_state = self.states[self.current_index]
        next_state = self.states[self.current_index + 1]
        
        self.current_index += 1
        
        return next_state, current_state

    def _close_position(self, reward: float, net_profit: float):
        """Close the current position and record the trade"""
        if self.entry_time and self.exit_time:
            trade_record = [
                self.entry_id,
                self.entry_time,
                self.entry_price,
                self.exit_id,
                self.exit_time,
                self.exit_price,
                net_profit,
                self.last_position
            ]
            self.trades.append(trade_record)
            
            # Reset position tracking
            self.entry_time = None
            self.entry_id = None
            self.entry_price = None
            self.exit_time = None
            self.exit_id = None
            self.exit_price = None

    def reset(self, e=None):
        """Reset the environment to initial state"""
        self.done = False
        self.current_index = 0

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

        self.total_reward = 0
        self.total_net_profit = 0
        self.orders = []
        self.trades = []
        self.feature_data = []

        if self.states:
            # Return the first state object directly for consistency with step()
            initial_state = self.states[0]
            return initial_state
        
        # Return empty state if no states available
        return None

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Episode: {self.episode}")
            print(f"Current Index: {self.current_index}")
            print(f"Current Position: {self.current_position}")
            print(f"Total Reward: {self.total_reward}")
            print(f"Total Trades: {len(self.trades)}")
            if self.trades:
                profitable_trades = sum(1 for trade in self.trades if trade[6] > 0)
                print(f"Profitable Trades: {profitable_trades}/{len(self.trades)}")
