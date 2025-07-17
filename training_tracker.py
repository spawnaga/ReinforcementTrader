"""
Professional Training Tracker for PostgreSQL
Tracks all aspects of RL training for analysis
"""
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from logging_config import get_loggers

class TrainingTracker:
    def __init__(self, session_id: str, algorithm_name: str, ticker: str, hyperparameters: Dict):
        self.session_id = session_id
        self.algorithm_name = algorithm_name
        self.ticker = ticker
        self.conn = self._get_connection()
        self.loggers = get_loggers()
        
        # Initialize session in database
        self._create_session(hyperparameters)
        
        # Track current state
        self.current_episode = 0
        self.current_trade_number = 0
        self.episode_trades = []
        self.episode_start_time = None
        
    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(os.environ['DATABASE_URL'])
        
    def _create_session(self, hyperparameters: Dict):
        """Create new training session in database"""
        with self.conn.cursor() as cur:
            # Get GPU info
            try:
                import torch
                gpu_info = {
                    'cuda_available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'device_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
                }
            except:
                gpu_info = {'cuda_available': False}
                
            cur.execute("""
                INSERT INTO training_sessions (session_id, algorithm_name, ticker, hyperparameters, gpu_info)
                VALUES (%s, %s, %s, %s, %s)
            """, (self.session_id, self.algorithm_name, self.ticker, 
                  json.dumps(hyperparameters), json.dumps(gpu_info)))
            self.conn.commit()
            
        # Log session start
        self.loggers['algorithm'].info(f"Training session started: {self.session_id}")
        self.loggers['algorithm'].info(f"Algorithm: {self.algorithm_name}, Ticker: {self.ticker}")
        self.loggers['algorithm'].info(f"Hyperparameters: {hyperparameters}")
        
    def start_episode(self, episode_number: int):
        """Mark start of new episode"""
        self.current_episode = episode_number
        self.episode_start_time = datetime.now()
        self.episode_trades = []
        self.current_trade_number = 0
        
        self.loggers['algorithm'].info(f"Episode {episode_number} started")
        
    def log_trade(self, entry_time: datetime, exit_time: datetime, position_type: str,
                  entry_price: float, exit_price: float, profit: float, 
                  commission: float, slippage: float, exit_reason: str = 'signal',
                  state_features: Optional[Dict] = None):
        """Log individual trade to database and files"""
        self.current_trade_number += 1
        
        # Calculate metrics
        if position_type == 'LONG':
            profit_ticks = (exit_price - entry_price) * 4  # NQ = 4 ticks per point
        else:  # SHORT
            profit_ticks = (entry_price - exit_price) * 4
            
        net_profit = profit - commission - slippage
        duration_minutes = int((exit_time - entry_time).total_seconds() / 60)
        
        # Store in database
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO trades (
                    session_id, episode_number, trade_number, entry_time, exit_time,
                    position_type, entry_price, exit_price, gross_profit, commission,
                    slippage, net_profit, profit_ticks, trade_duration_minutes,
                    exit_reason, state_features
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, self.current_episode, self.current_trade_number,
                entry_time, exit_time, position_type, entry_price, exit_price,
                profit, commission, slippage, net_profit, profit_ticks,
                duration_minutes, exit_reason, json.dumps(state_features) if state_features else None
            ))
            self.conn.commit()
            
        # Log to files
        self.loggers['trading'].info(
            f"Trade #{self.current_trade_number} | {position_type} | "
            f"Entry: ${entry_price:.2f} @ {entry_time.strftime('%H:%M:%S')} | "
            f"Exit: ${exit_price:.2f} @ {exit_time.strftime('%H:%M:%S')} | "
            f"Net P/L: ${net_profit:.2f} ({profit_ticks:.1f} ticks)"
        )
        
        self.loggers['rewards'].info(
            f"Episode {self.current_episode} Trade {self.current_trade_number}: "
            f"${net_profit:.2f} | Cumulative: ${sum(t['net_profit'] for t in self.episode_trades):.2f}"
        )
        
        # Track for episode summary
        self.episode_trades.append({
            'net_profit': net_profit,
            'profit_ticks': profit_ticks,
            'duration': duration_minutes
        })
        
    def log_position(self, timestamp: datetime, position: int, current_price: float,
                     unrealized_pnl: float, account_value: float):
        """Log position snapshot"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO position_snapshots (
                    session_id, timestamp, episode_number, position,
                    current_price, unrealized_pnl, account_value
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, timestamp, self.current_episode,
                position, current_price, unrealized_pnl, account_value
            ))
            self.conn.commit()
            
        self.loggers['positions'].info(
            f"{timestamp.strftime('%H:%M:%S')} | Position: {position:+d} | "
            f"Price: ${current_price:.2f} | Unrealized: ${unrealized_pnl:+.2f} | "
            f"Account: ${account_value:,.2f}"
        )
        
    def log_algorithm_decision(self, step: int, action: str, action_probs: Dict,
                              reward: float, q_values: Optional[Dict] = None,
                              policy_loss: Optional[float] = None,
                              value_loss: Optional[float] = None):
        """Log algorithm decision and metrics"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO algorithm_metrics (
                    session_id, episode_number, step_number, action_taken,
                    action_probabilities, q_values, policy_loss, value_loss, reward
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, self.current_episode, step, action,
                json.dumps(action_probs), json.dumps(q_values) if q_values else None,
                policy_loss, value_loss, reward
            ))
            self.conn.commit()
            
    def end_episode(self, total_reward: float, steps: int):
        """Complete episode and calculate metrics"""
        if not self.episode_start_time:
            return
            
        end_time = datetime.now()
        
        # Calculate episode metrics
        total_profit = sum(t['net_profit'] for t in self.episode_trades)
        num_trades = len(self.episode_trades)
        winning_trades = sum(1 for t in self.episode_trades if t['net_profit'] > 0)
        losing_trades = sum(1 for t in self.episode_trades if t['net_profit'] < 0)
        
        # Calculate performance metrics
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        avg_duration = np.mean([t['duration'] for t in self.episode_trades]) if num_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if num_trades > 1:
            returns = [t['net_profit'] for t in self.episode_trades]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
            
        # Store in database
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO episode_metrics (
                    session_id, episode_number, total_reward, total_profit,
                    num_trades, num_winning_trades, num_losing_trades,
                    sharpe_ratio, win_rate, avg_trade_duration_minutes,
                    start_time, end_time, steps
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, self.current_episode, total_reward, total_profit,
                num_trades, winning_trades, losing_trades, sharpe_ratio, win_rate,
                int(avg_duration), self.episode_start_time, end_time, steps
            ))
            self.conn.commit()
            
        # Log episode summary
        self.loggers['performance'].info(
            f"Episode {self.current_episode} | Reward: {total_reward:.2f} | "
            f"Profit: ${total_profit:.2f} | Trades: {num_trades} | "
            f"Win Rate: {win_rate:.2%} | Sharpe: {sharpe_ratio:.2f}"
        )
        
        # Update learning progress
        self._update_learning_progress(total_reward, total_profit, sharpe_ratio, win_rate)
        
    def _update_learning_progress(self, reward: float, profit: float, 
                                 sharpe: float, win_rate: float):
        """Update rolling metrics for learning curve"""
        with self.conn.cursor() as cur:
            # Get last 100 episodes for rolling average
            cur.execute("""
                SELECT total_reward, total_profit, sharpe_ratio, win_rate
                FROM episode_metrics
                WHERE session_id = %s
                ORDER BY episode_number DESC
                LIMIT 100
            """, (self.session_id,))
            
            recent_episodes = cur.fetchall()
            
            if recent_episodes:
                rolling_reward = np.mean([e['total_reward'] for e in recent_episodes])
                rolling_profit = np.mean([e['total_profit'] for e in recent_episodes])
                rolling_sharpe = np.mean([e['sharpe_ratio'] for e in recent_episodes])
                rolling_win_rate = np.mean([e['win_rate'] for e in recent_episodes])
            else:
                rolling_reward = reward
                rolling_profit = profit
                rolling_sharpe = sharpe
                rolling_win_rate = win_rate
                
            # Store learning progress
            cur.execute("""
                INSERT INTO learning_progress (
                    session_id, episode_number, rolling_avg_reward,
                    rolling_avg_profit, rolling_sharpe_ratio, rolling_win_rate
                ) VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, self.current_episode, rolling_reward,
                rolling_profit, rolling_sharpe, rolling_win_rate
            ))
            self.conn.commit()
            
    def save_checkpoint(self, model_path: str, episode: int, avg_reward: float,
                       best_reward: float, is_best: bool = False):
        """Save model checkpoint info"""
        checkpoint_name = f"{self.algorithm_name}_{self.ticker}_ep{episode}_{self.session_id}"
        
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_checkpoints (
                    session_id, checkpoint_name, episode_number,
                    avg_reward, best_reward, model_path, is_best
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                self.session_id, checkpoint_name, episode,
                avg_reward, best_reward, model_path, is_best
            ))
            self.conn.commit()
            
        self.loggers['algorithm'].info(
            f"Model checkpoint saved: {checkpoint_name} | "
            f"Avg Reward: {avg_reward:.2f} | Best: {is_best}"
        )
        
    def get_learning_assessment(self) -> Dict:
        """Quick assessment if agent is learning"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Get recent vs early performance
            cur.execute("""
                WITH early_episodes AS (
                    SELECT AVG(total_reward) as early_reward,
                           AVG(total_profit) as early_profit,
                           AVG(win_rate) as early_win_rate
                    FROM episode_metrics
                    WHERE session_id = %s AND episode_number <= 10
                ),
                recent_episodes AS (
                    SELECT AVG(total_reward) as recent_reward,
                           AVG(total_profit) as recent_profit,
                           AVG(win_rate) as recent_win_rate
                    FROM episode_metrics
                    WHERE session_id = %s AND episode_number > (
                        SELECT MAX(episode_number) - 10 
                        FROM episode_metrics 
                        WHERE session_id = %s
                    )
                )
                SELECT * FROM early_episodes, recent_episodes
            """, (self.session_id, self.session_id, self.session_id))
            
            comparison = cur.fetchone()
            
            if comparison and comparison['early_reward'] is not None:
                return {
                    'is_learning': comparison['recent_reward'] > comparison['early_reward'],
                    'reward_improvement': (comparison['recent_reward'] - comparison['early_reward']) / (abs(comparison['early_reward']) + 1e-6),
                    'profit_improvement': (comparison['recent_profit'] - comparison['early_profit']) / (abs(comparison['early_profit']) + 1e-6),
                    'win_rate_improvement': comparison['recent_win_rate'] - comparison['early_win_rate'],
                    'early_metrics': {
                        'reward': comparison['early_reward'],
                        'profit': comparison['early_profit'],
                        'win_rate': comparison['early_win_rate']
                    },
                    'recent_metrics': {
                        'reward': comparison['recent_reward'],
                        'profit': comparison['recent_profit'],
                        'win_rate': comparison['recent_win_rate']
                    }
                }
            else:
                return {'is_learning': False, 'message': 'Not enough episodes yet'}
                
    def close(self):
        """Close database connection"""
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE training_sessions 
                SET end_time = CURRENT_TIMESTAMP, status = 'completed'
                WHERE session_id = %s
            """, (self.session_id,))
            self.conn.commit()
        self.conn.close()