import os
import threading
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from flask import current_app

from models import TradingSession, Trade, MarketData, TrainingMetrics
from gym_futures.envs.futures_env import FuturesEnv
from gym_futures.envs.utils import TimeSeriesState
from rl_algorithms.ane_ppo import ANEPPO
from rl_algorithms.genetic_optimizer import GeneticOptimizer
from data_manager import DataManager
from ib_integration import IBIntegration
from risk_manager import RiskManager
from db_utils import retry_on_db_error, check_and_fix_db_permissions, ensure_db_writable

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
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # Use first GPU
            logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s) detected")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        logger.info(f"Trading Engine initialized with device: {self.device}")
        
        # Initialize default algorithm configurations
        self._initialize_default_configs()
    
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
                'genetic_population_size': 50,
                'genetic_mutation_rate': 0.1,
                'genetic_crossover_rate': 0.8,
                'attention_heads': 8,
                'attention_dim': 256,
                'transformer_layers': 6
            }
            
            # Market parameters for NQ futures
            self.market_params = {
                'tick_size': 0.25,
                'value_per_tick': 5.0,
                'execution_cost_per_order': 2.50,
                'fill_probability': 0.98,
                'long_values': [-2, -1, 0, 1, 2],
                'long_probabilities': [0.05, 0.15, 0.6, 0.15, 0.05],
                'short_values': [-2, -1, 0, 1, 2],
                'short_probabilities': [0.05, 0.15, 0.6, 0.15, 0.05]
            }
            
            logger.info("Default configurations initialized")
            
        except Exception as e:
            logger.error(f"Error initializing default configs: {str(e)}")
    
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
    
    def _training_loop(self, session_id: int, config: Dict):
        """Main training loop for a session"""
        from app import app
        
        # Create app context for the entire training thread
        with app.app_context():
            try:
                logger.info(f"Starting training loop for session {session_id}")
                # Update session status
                self.active_sessions[session_id]['status'] = 'running'
                
                # Load market data
                logger.info(f"Loading NQ market data for session {session_id}")
                market_data = self.data_manager.load_nq_data()
                if market_data is None or len(market_data) == 0:
                    logger.error(f"No market data available for session {session_id}")
                    self._end_session(session_id, 'error')
                    return
                
                # Limit data for reasonable training time
                original_size = len(market_data)
                max_rows = 50000
                if len(market_data) > max_rows:
                    market_data = market_data.tail(max_rows)
                    logger.info(f"Limited data from {original_size:,} to {max_rows:,} rows for training")
                
                logger.info(f"Using {len(market_data):,} rows of market data for session {session_id}")
                logger.debug(f"Market data columns: {list(market_data.columns)}")
                logger.debug(f"Market data shape: {market_data.shape}")
                logger.debug(f"First few rows:\n{market_data.head()}")
                
                # Create time series states
                logger.info(f"Creating time series states for session {session_id}")
                states = self._create_time_series_states(market_data)
                if not states:
                    logger.error(f"Failed to create any time series states for session {session_id}")
                    self._end_session(session_id, 'error')
                    return
                logger.info(f"Created {len(states)} time series states for session {session_id}")
                
                # Create trading environment
                logger.info(f"Creating FuturesEnv for session {session_id}")
                env = FuturesEnv(
                    states=states,
                    value_per_tick=self.market_params['value_per_tick'],
                    tick_size=self.market_params['tick_size'],
                    fill_probability=self.market_params['fill_probability'],
                    long_values=self.market_params['long_values'],
                    long_probabilities=self.market_params['long_probabilities'],
                    short_values=self.market_params['short_values'],
                    short_probabilities=self.market_params['short_probabilities'],
                    execution_cost_per_order=self.market_params['execution_cost_per_order']
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
                    from extensions import socketio
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
            
            # Limit the number of states for testing
            max_states = min(100, len(market_data) - window_size)  # Only create 100 states for testing
            
            logger.info(f"Will create {max_states} states from the data")
            
            for i in range(window_size, window_size + max_states):
                if i % 10 == 0:  # Log progress every 10 states  
                    logger.info(f"Creating state {i-window_size+1}/{max_states} (progress: {((i-window_size+1)/max_states)*100:.1f}%)")
                
                # Get window of data
                window_data = market_data.iloc[i-window_size:i].copy()
                
                # Add technical indicators
                logger.debug(f"Adding technical indicators to window {i-window_size+1}")
                window_data = self._add_technical_indicators(window_data)
                
                # Ensure numeric columns are properly typed
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in window_data.columns:
                        window_data[col] = pd.to_numeric(window_data[col], errors='coerce')
                
                # Remove any rows with NaN values in critical columns
                window_data = window_data.dropna(subset=['close'])
                
                if len(window_data) == 0:
                    logger.warning(f"Window {i-window_size+1} has no valid data after cleaning")
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
                            logger.error(f"No timestamp column found in data. Columns: {list(window_data.columns)}")
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
                    
                    logger.debug(f"State created successfully for window {i-window_size+1}, price: {state.price}")
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
            
            # Fill NaN values
            data.ffill(inplace=True)
            data.fillna(0, inplace=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {str(e)}")
            return data
    
    def _create_algorithm(self, algorithm_type: str, env, config: Dict):
        """Create the specified algorithm"""
        try:
            if algorithm_type == 'ANE_PPO':
                return ANEPPO(
                    env=env,
                    device=self.device,
                    **config.get('parameters', {})
                )
            else:
                logger.warning(f"Unknown algorithm type: {algorithm_type}, defaulting to ANE_PPO")
                return ANEPPO(
                    env=env,
                    device=self.device,
                    **config.get('parameters', {})
                )
                
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
                    logger.warning(f"Received None reward in episode {episode}, step {step_count}")
                    reward = 0
                
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
            
            episode_metrics = {
                'steps': step_count,
                'action_distribution': action_distribution,
                'total_trades': len(env.trades),
                'profitable_trades': len([t for t in env.trades if t[6] > 0]),  # Assuming profit is at index 6
                'average_trade_profit': np.mean([t[6] for t in env.trades]) if env.trades else 0
            }
            
            # Save training metrics to database
            self._save_training_metrics(session_id, episode, episode_reward, episode_loss, episode_metrics)
            
            return episode_reward, episode_loss, episode_metrics
            
        except Exception as e:
            logger.error(f"Error training episode {episode}: {str(e)}")
            return 0, 0, {}
    
    @retry_on_db_error(max_retries=3, delay=1.0)
    def _save_training_metrics(self, session_id: int, episode: int, reward: float, loss: float, metrics: Dict):
        """Save training metrics to database with retry logic"""
        from extensions import db
        from app import app
        try:
            with app.app_context():
                # Ensure database is writable
                ensure_db_writable()
                
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
        from extensions import db
        from app import app
        try:
            with app.app_context():
                # Ensure database is writable
                ensure_db_writable()
                
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
        from extensions import db, socketio
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
    
    def get_active_sessions(self) -> Dict:
        """Get all active training sessions"""
        # Clean up any stale sessions before returning
        self._cleanup_stale_sessions()
        return self.active_sessions.copy()
    
    def _cleanup_stale_sessions(self):
        """Remove any sessions that are no longer active"""
        stale_sessions = []
        for session_id, session_info in self.active_sessions.items():
            # Check if thread is still alive
            if session_id in self.training_threads:
                thread = self.training_threads[session_id]
                if not thread.is_alive():
                    stale_sessions.append(session_id)
            else:
                # No thread means session is stale
                stale_sessions.append(session_id)
        
        # Remove stale sessions
        for session_id in stale_sessions:
            logger.info(f"Cleaning up stale session {session_id}")
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            if session_id in self.training_threads:
                del self.training_threads[session_id]
