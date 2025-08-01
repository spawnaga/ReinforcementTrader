import os
import threading
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from flask import current_app

from app import db
from models import TradingSession, Trade, MarketData, TrainingMetrics
from gym_futures.envs.futures_env import FuturesEnv
from futures_env_realistic import RealisticFuturesEnv
from gym_futures.envs.utils import TimeSeriesState
from rl_algorithms.ane_ppo import ANEPPO
from rl_algorithms.genetic_optimizer import GeneticOptimizer
from data_manager import DataManager
from ib_integration import IBIntegration
from risk_manager import RiskManager
from db_utils import retry_on_db_error

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    Revolutionary GPU-accelerated trading engine with hybrid RL algorithms
    """

    def __init__(self):
        self.active_sessions = {}
        self.training_threads = {}
        self.data_manager = DataManager()
        self.ib_integration = IBIntegration()
        self.risk_manager = RiskManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_count = 0
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available: {self.gpu_count} GPU(s) detected")
            for i in range(self.gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            # Use all GPUs by default with DataParallel
            if self.gpu_count > 1:
                logger.info(f"Multi-GPU training enabled with {self.gpu_count} GPUs")

        logger.info(f"Trading Engine initialized with device: {self.device}")

        # Initialize default algorithm configurations
        self._initialize_default_configs()
        
        # Synchronize with database sessions on startup
        self._sync_database_sessions()

    def _initialize_default_configs(self):
        """Initialize default algorithm configurations"""
        try:
            # ANE-PPO Configuration
            ane_ppo_config = {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.5,
                'max_grad_norm': 0.5,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'genetic_population_size': 100,  # Increased from 50 for better diversity
                'genetic_mutation_rate': 0.1,
                'genetic_crossover_rate': 0.8,
                'attention_heads': 8,
                'attention_dim': 256,
                'transformer_layers': 6,
                'convergence_generations': 5  # Added for tighter convergence criterion
            }

            # Market parameters for NQ futures
            self.market_params = {
                'tick_size': 0.25,
                'value_per_tick': 5.0,
                'execution_cost_per_order': 5.0,  # Adjusted to match log examples
                'fill_probability': 0.98,
                'long_values': [-2, -1, 0, 1, 2],
                'long_probabilities': [0.05, 0.15, 0.6, 0.15, 0.05],
                'short_values': [-2, -1, 0, 1, 2],
                'short_probabilities': [0.05, 0.15, 0.6, 0.15, 0.05]
            }

            logger.info("Default configurations initialized")

        except Exception as e:
            logger.error(f"Error initializing default configs: {str(e)}")
    
    def _sync_database_sessions(self):
        """Synchronize with database sessions on startup"""
        try:
            from app import app
            with app.app_context():
                # Find all active sessions in database
                active_db_sessions = TradingSession.query.filter_by(status='active').all()
                
                for session in active_db_sessions:
                    # Add to active_sessions dictionary but without running thread
                    # This allows monitoring to detect them
                    self.active_sessions[session.id] = {
                        'status': 'active',
                        'config': {
                            'algorithm_type': session.algorithm_type,
                            'total_episodes': session.total_episodes
                        },
                        'start_time': session.start_time,
                        'current_episode': session.current_episode
                    }
                    logger.info(f"Synchronized with existing active session {session.id}")
                
                if active_db_sessions:
                    logger.info(f"Synchronized {len(active_db_sessions)} active sessions from database")
                else:
                    logger.info("No active sessions found in database")
                    
        except Exception as e:
            logger.error(f"Error synchronizing database sessions: {str(e)}")

    def start_training(self, session_id: int, config: Dict):
        """Start a new training session"""
        try:
            # Check if any session is already running to prevent database locks
            active_count = sum(1 for s in self.active_sessions.values()
                               if s.get('status') in ['starting', 'running'])
            if active_count > 0:
                logger.warning(f"Another training session is already active. Please wait for it to complete.")
                return False

            if session_id in self.active_sessions:
                logger.warning(f"Session {session_id} already active")
                return False

            # Create training thread
            training_thread = threading.Thread(
                target=self._training_loop,
                args=(session_id, config),
                daemon=True
            )

            self.training_threads[session_id] = training_thread
            self.active_sessions[session_id] = {
                'status': 'starting',
                'config': config,
                'start_time': datetime.utcnow()
            }

            training_thread.start()
            logger.info(f"Training session {session_id} started")
            
            # Immediately broadcast that we have a new active session
            try:
                from app import socketio
                
                # Emit active sessions count
                socketio.emit('active_sessions_count', {
                    'count': len(self.active_sessions),
                    'session_ids': list(self.active_sessions.keys())
                })
                
                # Emit training started event for advanced dashboard integration
                # Get session name from database
                session_name = f"Session_{session_id}"
                try:
                    session = db.session.get(TradingSession, session_id)
                    if session:
                        session_name = session.session_name
                except Exception as e:
                    logger.error(f"Failed to get session name: {e}")
                
                socketio.emit('training_started', {
                    'session_id': session_id,
                    'session_name': session_name,
                    'algorithm': config.get('algorithm_type', 'ANE_PPO').lower(),
                    'total_episodes': config.get('total_episodes', 1000),
                    'parameters': config.get('parameters', {})
                })
                
                logger.info(f"Broadcast training started for session {session_id}")
            except Exception as e:
                logger.error(f"Failed to broadcast active sessions: {e}")

            return True

        except Exception as e:
            logger.error(f"Error starting training session {session_id}: {str(e)}")
            return False

    def stop_training(self, session_id: int):
        """Stop a training session"""
        try:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'stopping'
                logger.info(f"Training session {session_id} marked for stopping")
                return True
            else:
                logger.warning(f"Session {session_id} not found in active sessions")
                return False

        except Exception as e:
            logger.error(f"Error stopping training session {session_id}: {str(e)}")
            return False
    
    def clear_all_sessions(self):
        """Clear all active sessions - used for recovery from stuck states"""
        try:
            count = len(self.active_sessions)
            self.active_sessions.clear()
            self.training_threads.clear()
            logger.info(f"Cleared {count} active sessions")
            return True
        except Exception as e:
            logger.error(f"Error clearing sessions: {str(e)}")
            return False

    def _training_loop(self, session_id: int, config: Dict):
        """Main training loop for a session"""
        from app import app

        # Create app context for the entire training thread
        with app.app_context():
            try:
                logger.info(f"Starting training loop for session {session_id}")
                # Update session status
                self.active_sessions[session_id]['status'] = 'running'

                # Load market data with strict limits
                logger.info(f"Loading limited NQ market data for session {session_id}")
                
                # Check if we have custom data for this session
                if self.active_sessions[session_id].get('use_custom_data'):
                    logger.info(f"Using custom data for session {session_id}")
                    market_data = self.active_sessions[session_id].get('training_data')
                    if market_data is None:
                        logger.error(f"No custom data found for session {session_id}")
                        self._end_session(session_id, 'error')
                        return
                else:
                    # Load from database which has 7,406 records
                    try:
                        from sqlalchemy import text
                        with db.engine.connect() as conn:
                            # First check how many records we have
                            count_query = text("SELECT COUNT(*) FROM market_data WHERE symbol = 'NQ'")
                            total_count = conn.execute(count_query).scalar()
                            logger.info(f"Database has {total_count} total NQ records")
                            
                            # Get data configuration from session
                            data_config = config.get('parameters', {}).get('dataConfig', {})
                            range_type = data_config.get('type', 'all')
                            logger.info(f"Data configuration: {data_config}")
                            
                            # Build query based on data configuration
                            if range_type == 'percentage':
                                percentage = data_config.get('percentage', 100)
                                limit = int(total_count * percentage / 100)
                                limit = min(limit, 500)  # Cap at 500 for memory
                                query = text(f"""
                                    SELECT * FROM market_data 
                                    WHERE symbol = 'NQ' 
                                    ORDER BY timestamp DESC 
                                    LIMIT {limit}
                                """)
                                market_data = pd.read_sql(query, conn)
                            
                            elif range_type == 'daterange':
                                start_date = data_config.get('startDate')
                                end_date = data_config.get('endDate')
                                query = text("""
                                    SELECT * FROM market_data 
                                    WHERE symbol = 'NQ' 
                                    AND timestamp >= :start_date 
                                    AND timestamp <= :end_date
                                    ORDER BY timestamp
                                    LIMIT 500
                                """)
                                market_data = pd.read_sql(query, conn, params={
                                    'start_date': start_date,
                                    'end_date': end_date
                                })
                            
                            elif range_type == 'timeperiod':
                                period = data_config.get('period', 'all')
                                if period != 'all':
                                    from datetime import datetime, timedelta
                                    period_map = {
                                        '1month': 30, '3months': 90, '6months': 180,
                                        '1year': 365, '2years': 730, '3years': 1095,
                                        '5years': 1825, '10years': 3650
                                    }
                                    days = period_map.get(period, 365)
                                    end_date = datetime.now()
                                    start_date = end_date - timedelta(days=days)
                                    
                                    query = text("""
                                        SELECT * FROM market_data 
                                        WHERE symbol = 'NQ' 
                                        AND timestamp >= :start_date
                                        ORDER BY timestamp
                                        LIMIT 500
                                    """)
                                    market_data = pd.read_sql(query, conn, params={
                                        'start_date': start_date.strftime('%Y-%m-%d')
                                    })
                                else:
                                    # Load all data but with limit
                                    limit = min(total_count, 500)
                                    query = text(f"""
                                        SELECT * FROM market_data 
                                        WHERE symbol = 'NQ' 
                                        ORDER BY timestamp DESC 
                                        LIMIT {limit}
                                    """)
                                    market_data = pd.read_sql(query, conn)
                            else:
                                # Default: load limited data
                                limit = min(total_count, 500)
                                query = text(f"""
                                    SELECT * FROM market_data 
                                    WHERE symbol = 'NQ' 
                                    ORDER BY timestamp DESC 
                                    LIMIT {limit}
                                """)
                                market_data = pd.read_sql(query, conn)
                        
                        if market_data is None or len(market_data) == 0:
                            logger.error(f"No market data available in database for session {session_id}")
                            self._end_session(session_id, 'error')
                            return
                            
                        # Sort by timestamp ascending after limiting
                        market_data = market_data.sort_values('timestamp')
                        
                        # Rename columns to match what the rest of the code expects
                        market_data = market_data.rename(columns={
                            'open_price': 'open',
                            'high_price': 'high',
                            'low_price': 'low',
                            'close_price': 'close'
                        })
                        
                        logger.info(f"Loaded {len(market_data)} rows from database")
                        
                    except Exception as e:
                        logger.error(f"Error loading market data: {str(e)}")
                        self._end_session(session_id, 'error')
                        return

                logger.info(f"Using {len(market_data):,} rows of market data for session {session_id}")
                logger.debug(f"Market data columns: {list(market_data.columns)}")
                logger.debug(f"Market data shape: {market_data.shape}")
                logger.debug(f"First few rows:\n{market_data.head()}")
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Memory cleaned up before state creation")

                # Create time series states
                logger.info(f"Creating time series states for session {session_id}")
                states = self._create_time_series_states(market_data)
                if not states:
                    logger.error(f"Failed to create any time series states for session {session_id}")
                    self._end_session(session_id, 'error')
                    return
                logger.info(f"Created {len(states)} time series states for session {session_id}")

                # Create trading environment with realistic constraints
                logger.info(f"Creating RealisticFuturesEnv for session {session_id}")
                env = RealisticFuturesEnv(
                    states=states,
                    value_per_tick=self.market_params['value_per_tick'],
                    tick_size=self.market_params['tick_size'],
                    fill_probability=self.market_params.get('fill_probability', 0.95),
                    execution_cost_per_order=self.market_params.get('execution_cost_per_order', 5.0),
                    min_holding_periods=10,  # Must hold for 10 time steps
                    max_trades_per_episode=5,  # Max 5 trades per episode
                    slippage_ticks=2,  # Average 2 tick slippage
                    session_id=session_id
                )
                logger.info(f"FuturesEnv created successfully for session {session_id}")

                # Initialize algorithm
                algorithm_type = config.get('algorithm_type', 'ANE_PPO')
                logger.info(f"Creating {algorithm_type} algorithm for session {session_id}")
                algorithm = self._create_algorithm(algorithm_type, env, config)
                logger.info(f"Algorithm created successfully for session {session_id}")

                # Training parameters
                total_episodes = config.get('total_episodes', 1000)
                save_interval = config.get('save_interval', 100)

                # Training loop
                for episode in range(total_episodes):
                    if self.active_sessions[session_id]['status'] == 'stopping':
                        break

                    # Train one episode
                    episode_reward, episode_loss, episode_metrics = self._train_episode(
                        env, algorithm, episode, session_id
                    )

                    # Update session statistics
                    self._update_session_stats(session_id, episode, episode_reward, episode_metrics)

                    # Save model periodically
                    if episode % save_interval == 0:
                        self._save_model(session_id, algorithm, episode)

                    # Emit real-time updates
                    from app import socketio
                    socketio.emit('training_update', {
                        'session_id': session_id,
                        'episode': episode,
                        'reward': episode_reward,
                        'loss': episode_loss,
                        'metrics': episode_metrics
                    })

                    # Small delay to prevent overwhelming the system
                    time.sleep(0.01)

                # End session
                self._end_session(session_id, 'completed')

            except Exception as e:
                logger.error(f"Error in training loop for session {session_id}: {str(e)}")
                self._end_session(session_id, 'error')

    def _create_time_series_states(self, market_data: pd.DataFrame) -> List[TimeSeriesState]:
        """Create time series states from market data"""
        try:
            states = []
            window_size = 60  # 1 hour of 1-minute data

            logger.info(f"Creating states from {len(market_data):,} rows with window size {window_size}")

            # Check if we have enough data
            if len(market_data) < window_size + 1:
                logger.error(f"Not enough data to create states. Need at least {window_size + 1} rows, have {len(market_data)}")
                return []

            # Limit the number of states for testing
            max_states = min(10, len(market_data) - window_size)  # Reduced to 10 states for faster testing

            logger.info(f"Will create {max_states} states from the data")
            
            # Use evenly spaced indices instead of sequential to cover more of the data
            indices = np.linspace(window_size, len(market_data) - 1, max_states, dtype=int)

            for idx, i in enumerate(indices):
                if idx % 5 == 0:  # Log progress every 5 states
                    logger.info(
                        f"Creating state {idx + 1}/{max_states} (progress: {((idx + 1) / max_states) * 100:.1f}%)")

                # Get window of data
                window_data = market_data.iloc[i - window_size:i].copy()

                # Add technical indicators
                logger.debug(f"Adding technical indicators to window {i - window_size + 1}")
                window_data = self._add_technical_indicators(window_data)
                
                # Ensure we have a 'time' column for TimeSeriesState
                if 'time' not in window_data.columns:
                    if window_data.index.name == 'timestamp':
                        # Reset index to make timestamp a column named 'time'
                        window_data = window_data.reset_index()
                        window_data.rename(columns={'timestamp': 'time'}, inplace=True)
                    else:
                        # Create a time column from the index
                        window_data['time'] = window_data.index

                # Ensure numeric columns are properly typed
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in window_data.columns:
                        window_data[col] = pd.to_numeric(window_data[col], errors='coerce')

                # Remove any rows with NaN values in critical columns
                window_data = window_data.dropna(subset=['close'])

                if len(window_data) == 0:
                    logger.warning(f"Window {i - window_size + 1} has no valid data after cleaning")
                    continue

                # Create state
                try:
                    # Check if timestamp column exists, otherwise try other common names
                    if 'timestamp' in window_data.columns:
                        ts_col = 'timestamp'
                    elif 'time' in window_data.columns:
                        ts_col = 'time'
                    elif 'date' in window_data.columns:
                        ts_col = 'date'
                    else:
                        # Use index if it's a datetime index
                        if isinstance(window_data.index, pd.DatetimeIndex):
                            window_data['timestamp'] = window_data.index
                            ts_col = 'timestamp'
                        else:
                            logger.error(
                                f"No timestamp column found in data. Columns: {list(window_data.columns)}")
                            continue

                    logger.debug(f"Creating TimeSeriesState with timestamp column: {ts_col}")

                    # Debug logging
                    if len(window_data) > 0:
                        logger.debug(f"Close price (last row): {window_data['close'].iloc[-1]}")
                        logger.debug(f"Data shape: {window_data.shape}")

                    # Create proper TimeSeriesState object
                    state = TimeSeriesState(
                        data=window_data,
                        close_price_identifier='close',
                        timestamp_identifier=ts_col,
                        timestamp_format='%Y-%m-%d %H:%M:%S'
                    )

                    logger.debug(
                        f"State created successfully for window {i - window_size + 1}, price: {state.price}")
                    states.append(state)

                except Exception as e:
                    logger.error(f"Error creating state: {str(e)}")
                    logger.debug(f"Window data columns: {list(window_data.columns)}")
                    logger.debug(f"Window data shape: {window_data.shape}")
                    continue

            logger.info(f"Created {len(states)} time series states")
            return states

        except Exception as e:
            logger.error(f"Error creating time series states: {str(e)}")
            logger.exception(e)  # Log full traceback
            return []

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to market data"""
        try:
            # Moving averages
            data['sma_10'] = data['close'].rolling(window=10).mean()
            data['sma_20'] = data['close'].rolling(window=20).mean()
            data['ema_10'] = data['close'].ewm(span=10).mean()
            data['ema_20'] = data['close'].ewm(span=20).mean()

            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            data['macd'] = ema_12 - ema_26
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']

            # Bollinger Bands
            data['bb_middle'] = data['close'].rolling(window=20).mean()
            bb_std = data['close'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)

            # Volume indicators
            data['volume_sma'] = data['volume'].rolling(window=20).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']

            # Price action
            data['high_low_ratio'] = (data['high'] - data['low']) / data['close']
            data['open_close_ratio'] = (data['close'] - data['open']) / data['close']

            # Price patterns
            data['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low']) < 0.1).astype(int)
            data['hammer'] = ((data['low'] < data[['open', 'close']].min(axis=1)) &
                              (data['high'] - data[['open', 'close']].max(axis=1) < (
                                          data[['open', 'close']].max(axis=1) - data['low']) * 0.3)).astype(int)

            # Market structure
            data['higher_high'] = (data['high'] > data['high'].shift(1)).astype(int)
            data['lower_low'] = (data['low'] < data['low'].shift(1)).astype(int)

            # Time-based features
            # Check if index is datetime, if not try to find timestamp column
            if isinstance(data.index, pd.DatetimeIndex):
                data['hour'] = data.index.hour
                data['minute'] = data.index.minute
                data['day_of_week'] = data.index.dayofweek
            elif 'timestamp' in data.columns:
                # Convert timestamp column to datetime if needed
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data['hour'] = data['timestamp'].dt.hour
                data['minute'] = data['timestamp'].dt.minute
                data['day_of_week'] = data['timestamp'].dt.dayofweek
            elif 'time' in data.columns:
                # Convert time column to datetime if needed
                data['time'] = pd.to_datetime(data['time'])
                data['hour'] = data['time'].dt.hour
                data['minute'] = data['time'].dt.minute
                data['day_of_week'] = data['time'].dt.dayofweek
            else:
                # Skip time-based features if no timestamp available
                logger.warning("No timestamp index or column found, skipping time-based features")
                data['hour'] = 0
                data['minute'] = 0
                data['day_of_week'] = 0
            
            # Add session features only if hour column exists
            if 'hour' in data.columns:
                data['is_session_start'] = (data['hour'] == 17).astype(int)
                data['is_session_end'] = (data['hour'] == 16).astype(int)
            else:
                data['is_session_start'] = 0
                data['is_session_end'] = 0

            # Fill NaN values
            data.ffill(inplace=True)
            data.fillna(0, inplace=True)

            return data

        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data

    def _create_algorithm(self, algorithm_type: str, env, config: Dict):
        """Create the specified algorithm with multi-GPU support"""
        try:
            # Extract parameters and remove non-algorithm parameters
            params = config.get('parameters', {}).copy()
            # Remove parameters that are not part of ANE-PPO constructor
            params.pop('dataConfig', None)  # Remove dataConfig as it's only for data loading
            params.pop('indicators', None)  # Remove indicators as it's for data preprocessing
            params.pop('timeframe', None)  # Remove timeframe as it's for data selection
            params.pop('total_episodes', None)  # Remove total_episodes as it's stored separately
            params.pop('risk_level', None)  # Remove risk_level as it's not an algorithm parameter
            
            if algorithm_type == 'ANE_PPO':
                algorithm = ANEPPO(
                    env=env,
                    device=self.device,
                    **params
                )
                # Enable multi-GPU training if available
                if self.gpu_count > 1:
                    algorithm.enable_multi_gpu(self.gpu_count)
                return algorithm
            else:
                logger.warning(f"Unknown algorithm type: {algorithm_type}, defaulting to ANE_PPO")
                algorithm = ANEPPO(
                    env=env,
                    device=self.device,
                    **params)
                if self.gpu_count > 1:
                    algorithm.enable_multi_gpu(self.gpu_count)
                return algorithm

        except Exception as e:
            logger.error(f"Error creating algorithm: {str(e)}")
            raise

    def _train_episode(self, env, algorithm, episode: int, session_id: int) -> Tuple[float, float, Dict]:
        """Train one episode"""
        try:
            # Reset environment
            state = env.reset(episode)

            episode_reward = 0
            episode_loss = 0
            step_count = 0
            actions_taken = []

            done = False
            while not done:
                # Get action from algorithm
                action = algorithm.get_action(state)

                # Step environment
                next_state, reward, done, info = env.step(action)

                # Ensure reward is not None
                if reward is None:
                    logger.error(
                        f"None reward encountered in episode {episode}, step {step_count}. State info: {state}. Forcing default reward to 0.0")
                    reward = 0.0

                # Store experience
                algorithm.store_experience(state, action, reward, next_state, done)

                # Update counters
                episode_reward += reward
                actions_taken.append(action)
                step_count += 1

                state = next_state

                # Check if session should stop
                if self.active_sessions[session_id]['status'] == 'stopping':
                    done = True
                    break

            # Train algorithm
            if hasattr(algorithm, 'train'):
                episode_loss = algorithm.train()

            # Calculate metrics
            action_counts = np.bincount(actions_taken, minlength=3)
            action_distribution = (action_counts / len(actions_taken)).tolist()

            # Calculate trade metrics
            if hasattr(env, 'trading_logger') and env.trading_logger:
                trades = env.trading_logger.trades
                completed_trades = [t for t in trades if t.get('action') == 'EXIT' and t.get('profit_loss') is not None]
                episode_metrics = {
                    'steps': step_count,
                    'action_distribution': action_distribution,
                    'total_trades': len(completed_trades),
                    'profitable_trades': len([t for t in completed_trades if t.get('profit_loss', 0) > 0]),
                    'average_trade_profit': np.mean([t.get('profit_loss', 0) for t in completed_trades]) if completed_trades else 0
                }
            else:
                episode_metrics = {
                    'steps': step_count,
                    'action_distribution': action_distribution,
                    'total_trades': 0,
                    'profitable_trades': 0,
                    'average_trade_profit': 0
                }

            # Save trades from the episode to database
            self._save_episode_trades(session_id, env)
            
            # Save training metrics to database
            self._save_training_metrics(session_id, episode, episode_reward, episode_loss, episode_metrics)

            return episode_reward, episode_loss, episode_metrics

        except Exception as e:
            logger.error(f"Error training episode {episode}: {str(e)}")
            return 0, 0, {}

    @retry_on_db_error(max_retries=3, delay=1.0)
    def _save_episode_trades(self, session_id: int, env):
        """Save trades from the episode to database"""
        from app import db
        from app import app
        from models import Trade
        
        try:
            with app.app_context():
                # Check if env has a trading_logger with trades
                if hasattr(env, 'trading_logger') and env.trading_logger:
                    trade_count = 0
                    # Get current episode from session
                    session = TradingSession.query.get(session_id)
                    current_episode = session.current_episode if session else 1
                    for trade_info in env.trading_logger.trades:
                        # Only save completed trades (exits)
                        if trade_info.get('action') == 'EXIT' and trade_info.get('entry_price') and trade_info.get('exit_price'):
                            trade_count += 1
                            # Convert timestamp string back to datetime if needed
                            timestamp = trade_info.get('timestamp', datetime.now())
                            if isinstance(timestamp, str):
                                try:
                                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                except:
                                    timestamp = datetime.now()
                            
                            trade = Trade(
                                session_id=session_id,
                                trade_id=f"T{session_id}_{current_episode}_{trade_count}",
                                entry_time=timestamp,
                                exit_time=timestamp,
                                position_type=trade_info.get('position_type', '').lower(),
                                entry_price=float(trade_info.get('entry_price', 0)),
                                exit_price=float(trade_info.get('exit_price', 0)),
                                quantity=trade_info.get('quantity', 1),
                                profit_loss=float(trade_info.get('profit_loss', 0)) if trade_info.get('profit_loss') is not None else 0.0,
                                status='closed',
                                episode_number=current_episode
                            )
                            db.session.add(trade)
                            
                            # Emit trade update via WebSocket
                            try:
                                from app import socketio
                                socketio.emit('trade_update', {
                                    'id': trade.id,
                                    'session_id': session_id,
                                    'position_type': trade.position_type,
                                    'entry_price': trade.entry_price,
                                    'exit_price': trade.exit_price,
                                    'profit_loss': trade.profit_loss,
                                    'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                                    'quantity': trade.quantity,
                                    'episode_number': current_episode,
                                    'reward': trade_info.get('reward', 0)
                                }, room=f'session_{session_id}')
                                
                                # Also emit training metrics
                                socketio.emit('training_metrics', {
                                    'episode': current_episode,
                                    'reward': trade_info.get('reward', 0),
                                    'loss': 0,  # Will be updated when we have loss data
                                    'episode_profit': trade.profit_loss
                                })
                                
                                logger.debug(f"Emitted trade update for session {session_id}")
                            except Exception as e:
                                logger.error(f"Failed to emit trade update: {e}")
                    
                    db.session.commit()
                    logger.info(f"Saved {len(env.trading_logger.trades)} trades for session {session_id}")
                    
        except Exception as e:
            logger.error(f"Error saving episode trades: {str(e)}")
            raise  # Re-raise to trigger retry

    @retry_on_db_error(max_retries=3, delay=1.0)
    def _save_training_metrics(self, session_id: int, episode: int, reward: float, loss: float, metrics: Dict):
        """Save training metrics to database with retry logic"""
        from app import db
        from app import app
        try:
            with app.app_context():
                metric = TrainingMetrics(
                    session_id=session_id,
                    episode=episode,
                    reward=reward,
                    loss=loss,
                    action_distribution=metrics.get('action_distribution', []),
                    network_weights_summary={}  # Could add weight statistics here
                )

                db.session.add(metric)
                db.session.commit()

        except Exception as e:
            logger.error(f"Error saving training metrics: {str(e)}")
            raise  # Re-raise to trigger retry

    @retry_on_db_error(max_retries=3, delay=1.0)
    def _update_session_stats(self, session_id: int, episode: int, reward: float, metrics: Dict):
        """Update session statistics with retry logic"""
        from app import db
        from app import app
        try:
            with app.app_context():
                session = TradingSession.query.get(session_id)
                if session:
                    session.current_episode = episode
                    session.total_profit += reward

                    # Update other statistics based on metrics
                    if metrics.get('total_trades', 0) > 0:
                        session.total_trades = metrics['total_trades']
                        session.win_rate = metrics.get('profitable_trades', 0) / metrics['total_trades']

                    db.session.commit()

        except Exception as e:
            logger.error(f"Error updating session stats: {str(e)}")
            raise  # Re-raise to trigger retry

    def _save_model(self, session_id: int, algorithm, episode: int):
        """Save model checkpoint"""
        try:
            model_dir = f"models/session_{session_id}"
            os.makedirs(model_dir, exist_ok=True)

            if hasattr(algorithm, 'save'):
                algorithm.save(f"{model_dir}/episode_{episode}.pt")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def _end_session(self, session_id: int, status: str):
        """End a training session"""
        from app import db, socketio
        from app import app
        try:
            # Update database
            with app.app_context():
                session = TradingSession.query.get(session_id)
                if session:
                    session.status = status
                    session.end_time = datetime.utcnow()
                    db.session.commit()

            # Clean up
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            if session_id in self.training_threads:
                del self.training_threads[session_id]

            # Emit completion event
            socketio.emit('training_complete', {
                'session_id': session_id,
                'status': status
            })
            
            # Broadcast updated active sessions count
            socketio.emit('active_sessions_count', {
                'count': len(self.active_sessions),
                'session_ids': list(self.active_sessions.keys())
            })

            logger.info(f"Training session {session_id} ended with status: {status}")

        except Exception as e:
            logger.error(f"Error ending session {session_id}: {str(e)}")

    def connect_ib(self, host: str, port: int, client_id: int) -> bool:
        """Connect to Interactive Brokers"""
        try:
            return self.ib_integration.connect(host, port, client_id)
        except Exception as e:
            logger.error(f"Error connecting to IB: {str(e)}")
            return False

    def disconnect_ib(self):
        """Disconnect from Interactive Brokers"""
        try:
            self.ib_integration.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from IB: {str(e)}")

    def get_active_sessions(self) -> List[int]:
        """Get all active training session IDs"""
        # Clean up any stale sessions before returning
        self._cleanup_stale_sessions()
        return list(self.active_sessions.keys())

    def _cleanup_stale_sessions(self):
        """Remove any sessions that are no longer active"""
        stale_sessions = []
        for session_id, session_info in self.active_sessions.items():
            # Check if thread is still alive
            if session_id in self.training_threads:
                thread = self.training_threads[session_id]
                if not thread.is_alive():
                    stale_sessions.append(session_id)
            elif session_info.get('status') == 'active':
                # This is a synchronized session from database, check if it's still active in DB
                from app import app
                with app.app_context():
                    db_session = TradingSession.query.get(session_id)
                    if not db_session or db_session.status != 'active':
                        stale_sessions.append(session_id)
            else:
                # No thread and not marked as active means session is stale
                stale_sessions.append(session_id)

        # Remove stale sessions
        for session_id in stale_sessions:
            logger.info(f"Cleaning up stale session {session_id}")
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.training_threads:
                del self.training_threads[session_id]

    # ------------------------------------------------------------------
    # Simple helper methods for manual trades and optimization requests
    # ------------------------------------------------------------------

    def place_trade(self, trade_info: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder manual trade handler"""
        try:
            logger.info(f"Manual trade request: {trade_info}")
            trade_id = f"T{int(time.time()*1000)}"
            return {'success': True, 'trade_id': trade_id, 'executed_price': trade_info.get('price')}
        except Exception as e:
            logger.error(f"Place trade error: {e}")
            return {'success': False, 'error': str(e)}

    def optimize_hyperparameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a genetic optimizer for hyperparameters"""
        try:
            logger.info(f"Optimization request: {config}")
            base_params = config.get('parameters', {})
            performance = config.get('performance', 0.0)
            optimizer = GeneticOptimizer()
            best = optimizer.evolve(base_params, performance)
            return {'success': True, 'best_parameters': best}
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_training_with_data(self, session_id: int, training_data: pd.DataFrame, 
                                test_data: Optional[pd.DataFrame] = None, ticker: str = 'NQ') -> bool:
        """
        Start training with pre-loaded data
        
        Args:
            session_id: Session ID
            training_data: Training data DataFrame
            test_data: Optional test data DataFrame
            ticker: Ticker symbol
            
        Returns:
            True if training started successfully
        """
        try:
            if session_id not in self.active_sessions:
                logger.error(f"Session {session_id} not found")
                return False
                
            # Store data in session
            self.active_sessions[session_id]['training_data'] = training_data
            self.active_sessions[session_id]['test_data'] = test_data
            self.active_sessions[session_id]['ticker'] = ticker
            
            # Start training with custom data flag
            self.active_sessions[session_id]['use_custom_data'] = True
            
            # Get config from session
            config = self.active_sessions[session_id]['config']
            
            # Start the training loop
            return self.start_training(session_id, config)
            
        except Exception as e:
            logger.error(f"Error starting training with data: {str(e)}")
            return False
    
    def create_training_session(self, algorithm_type: str, config: dict) -> int:
        """
        Create a new training session with configuration
        
        Args:
            algorithm_type: Type of algorithm to use
            config: Algorithm configuration
            
        Returns:
            Session ID
        """
        try:
            from app import app, db
            from models import TradingSession
            
            with app.app_context():
                # Create session in database
                session = TradingSession(
                    session_name=f"{algorithm_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                    algorithm_type=algorithm_type,
                    parameters=config,
                    status='created',
                    total_episodes=config.get('episodes', 1000),
                    current_episode=0,
                    total_profit=0.0,
                    total_trades=0,
                    win_rate=0.0,
                    sharpe_ratio=0.0,
                    max_drawdown=0.0
                )
                db.session.add(session)
                db.session.commit()
                
                session_id = session.id
                
                # Add algorithm type to config
                config['algorithm_type'] = algorithm_type
                
                # Create in-memory session
                self.active_sessions[session_id] = {
                    'status': 'created',
                    'config': config,
                    'algorithm_type': algorithm_type,
                    'created_at': datetime.utcnow(),
                    'episode': 0,
                    'total_reward': 0.0,
                    'metrics': {},
                    'trades': []
                }
            
            logger.info(f"Created training session {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating training session: {str(e)}")
            raise