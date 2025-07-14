#!/usr/bin/env python3
"""
Enhanced training monitor with fixed display and detailed metrics
"""
import os
import sys
import time
import curses
from datetime import datetime
from app import app
from extensions import db
from models import TradingSession, Trade, TrainingMetrics
from sqlalchemy import text
import numpy as np

def calculate_sharpe_ratio(returns):
    """Calculate Sharpe ratio from returns"""
    if len(returns) < 2:
        return 0.0
    
    returns_array = np.array(returns)
    if returns_array.std() == 0:
        return 0.0
    
    # Annualized Sharpe (assuming 252 trading days)
    return np.sqrt(252) * (returns_array.mean() / returns_array.std())

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown percentage"""
    if len(equity_curve) < 2:
        return 0.0
    
    equity_array = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    return abs(drawdown.min() * 100)  # Return as percentage

class EnhancedMonitor:
    def __init__(self):
        self.stdscr = None
        self.last_update = {}
        
    def run(self, stdscr):
        """Main monitoring loop with fixed display"""
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking input
        stdscr.timeout(2000)  # Update every 2 seconds
        
        while True:
            try:
                # Clear screen
                stdscr.clear()
                
                # Get window size
                height, width = stdscr.getmaxyx()
                
                # Display header
                self._display_header(width)
                
                with app.app_context():
                    # Get active session
                    session = TradingSession.query.filter_by(status='active').first()
                    
                    if session:
                        self._display_session_info(session, width)
                        self._display_algorithm_info(session, width)
                        self._display_data_info(session, width)
                        self._display_training_progress(session, width)
                        self._display_performance_metrics(session, width)
                        self._display_recent_trades(session, width)
                        self._display_system_info(width)
                    else:
                        self._display_no_session(width)
                
                # Display footer
                self._display_footer(width, height)
                
                # Refresh display
                stdscr.refresh()
                
                # Check for quit command
                key = stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Display error
                stdscr.addstr(height-2, 0, f"Error: {str(e)}"[:width-1])
                stdscr.refresh()
                time.sleep(2)
    
    def _display_header(self, width):
        """Display header"""
        title = "🚀 Enhanced AI Trading Monitor"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.stdscr.addstr(0, 0, "=" * width)
        self.stdscr.addstr(1, (width - len(title)) // 2, title)
        self.stdscr.addstr(2, (width - len(timestamp)) // 2, timestamp)
        self.stdscr.addstr(3, 0, "=" * width)
    
    def _display_session_info(self, session, width):
        """Display session information"""
        y = 5
        self.stdscr.addstr(y, 0, f"📊 Session: {session.session_name} (ID: {session.id})")
        self.stdscr.addstr(y+1, 2, f"Status: {session.status}")
        self.stdscr.addstr(y+2, 2, f"Started: {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _display_algorithm_info(self, session, width):
        """Display algorithm details"""
        y = 9
        self.stdscr.addstr(y, 0, "🧠 Algorithm: ANE-PPO (Attention Network Enhanced PPO)")
        self.stdscr.addstr(y+1, 2, "• Transformer attention mechanisms for market regime detection")
        self.stdscr.addstr(y+2, 2, "• Multi-scale feature extraction (short/medium/long term)")
        self.stdscr.addstr(y+3, 2, "• Actor-Critic architecture with dueling Q-networks")
        self.stdscr.addstr(y+4, 2, "• Genetic algorithm for hyperparameter optimization")
    
    def _display_data_info(self, session, width):
        """Display data usage information"""
        y = 15
        
        # Get total data count
        total_records = db.session.execute(text("SELECT COUNT(*) FROM market_data WHERE symbol = 'NQ'")).scalar()
        
        self.stdscr.addstr(y, 0, "📈 Data Usage:")
        self.stdscr.addstr(y+1, 2, f"Total dataset: {total_records:,} records")
        self.stdscr.addstr(y+2, 2, f"Window size: 60 time steps")
        self.stdscr.addstr(y+3, 2, f"States per episode: 10")
        self.stdscr.addstr(y+4, 2, f"Sequential processing: Yes (no shuffle)")
    
    def _display_training_progress(self, session, width):
        """Display training progress"""
        y = 21
        
        # Progress bar
        progress = session.current_episode / session.total_episodes
        bar_width = width - 10
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        self.stdscr.addstr(y, 0, "🎯 Training Progress:")
        self.stdscr.addstr(y+1, 2, f"Episodes: {session.current_episode}/{session.total_episodes}")
        self.stdscr.addstr(y+2, 2, f"[{bar}] {progress*100:.1f}%")
        
        # Get latest training metrics
        latest_metric = TrainingMetrics.query.filter_by(
            session_id=session.id
        ).order_by(TrainingMetrics.id.desc()).first()
        
        if latest_metric:
            self.stdscr.addstr(y+3, 2, f"Latest Reward: {latest_metric.reward:.2f}")
            self.stdscr.addstr(y+4, 2, f"Latest Loss: {latest_metric.loss:.4f}")
    
    def _display_performance_metrics(self, session, width):
        """Display performance metrics with Sharpe and drawdown"""
        y = 27
        
        # Get all trades for this session
        trades = Trade.query.filter_by(session_id=session.id).all()
        closed_trades = [t for t in trades if t.exit_time is not None]
        
        # Calculate metrics
        total_profit = sum(t.profit_loss for t in closed_trades)
        win_rate = 0
        sharpe_ratio = 0
        max_drawdown = 0
        
        if closed_trades:
            wins = sum(1 for t in closed_trades if t.profit_loss > 0)
            win_rate = (wins / len(closed_trades)) * 100
            
            # Calculate returns for Sharpe
            returns = [t.profit_loss for t in closed_trades]
            sharpe_ratio = calculate_sharpe_ratio(returns)
            
            # Calculate equity curve for drawdown
            equity_curve = [10000]  # Starting capital
            for trade in closed_trades:
                equity_curve.append(equity_curve[-1] + trade.profit_loss)
            max_drawdown = calculate_max_drawdown(equity_curve)
        
        self.stdscr.addstr(y, 0, "💰 Performance Metrics:")
        self.stdscr.addstr(y+1, 2, f"Total Trades: {len(closed_trades)}")
        self.stdscr.addstr(y+2, 2, f"Total Profit: ${total_profit:,.2f}")
        self.stdscr.addstr(y+3, 2, f"Win Rate: {win_rate:.1f}%")
        self.stdscr.addstr(y+4, 2, f"Sharpe Ratio: {sharpe_ratio:.2f}")
        self.stdscr.addstr(y+5, 2, f"Max Drawdown: {max_drawdown:.1f}%")
    
    def _display_recent_trades(self, session, width):
        """Display recent trades"""
        y = 34
        
        # Get recent trades
        recent_trades = Trade.query.filter_by(
            session_id=session.id
        ).order_by(Trade.id.desc()).limit(5).all()
        
        self.stdscr.addstr(y, 0, "📊 Recent Trades:")
        
        for i, trade in enumerate(recent_trades):
            if trade.exit_time:
                trade_str = f"#{trade.id}: ${trade.entry_price:.2f} → ${trade.exit_price:.2f} P&L: ${trade.profit_loss:.2f}"
            else:
                trade_str = f"#{trade.id}: OPEN @ ${trade.entry_price:.2f}"
            self.stdscr.addstr(y+1+i, 2, trade_str[:width-3])
    
    def _display_system_info(self, width):
        """Display system information"""
        y = 41
        
        # Get GPU count
        import torch
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        self.stdscr.addstr(y, 0, f"💻 System: {gpu_count} GPUs active")
    
    def _display_no_session(self, width):
        """Display when no active session"""
        y = 10
        self.stdscr.addstr(y, (width - 30) // 2, "❌ No active training session")
        self.stdscr.addstr(y+2, (width - 40) // 2, "Start training with run_local.py")
    
    def _display_footer(self, width, height):
        """Display footer"""
        self.stdscr.addstr(height-2, 0, "=" * width)
        self.stdscr.addstr(height-1, 0, "Press 'q' to quit | Updates every 2 seconds")

def main():
    """Run the enhanced monitor"""
    monitor = EnhancedMonitor()
    try:
        curses.wrapper(monitor.run)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()