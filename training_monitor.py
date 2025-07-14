#!/usr/bin/env python3
"""
Advanced Training Progress Monitor for AI Trading System
Shows detailed training metrics, algorithm performance, and financial indicators
"""

import time
import requests
import socketio
from datetime import datetime
import argparse
import sys
import json
from collections import deque
import statistics
import subprocess

class TrainingMonitor:
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')
        self.sio = socketio.Client()
        self.setup_handlers()
        
        # Track metrics history
        self.profit_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        self.sharpe_history = deque(maxlen=20)
        self.drawdown_history = deque(maxlen=20)
        
        # Current session info
        self.current_session = None
        self.algorithm_stats = {}
        self.start_time = None
        self.trades_count = 0
        self.win_count = 0
        self.loss_count = 0
        
    def setup_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.sio.on('connect')
        def on_connect():
            print(f"✅ Connected to Trading System at {self.api_url}")
            
        @self.sio.on('training_started')
        def on_training_start(data):
            self.current_session = data
            self.start_time = datetime.now()
            print(f"\n🚀 Training Started: {data.get('session_name', 'Session')}")
            print(f"   Algorithm: {data.get('algorithm', 'ANE-PPO').upper()}")
            print(f"   Total Episodes: {data.get('total_episodes', 0)}")
            
        @self.sio.on('session_update')
        def on_session_update(data):
            if self.current_session:
                self.current_session.update(data)
                
        @self.sio.on('trade_update')
        def on_trade(data):
            self.trades_count += 1
            if data.get('profit_loss', 0) > 0:
                self.win_count += 1
            else:
                self.loss_count += 1
                
        @self.sio.on('training_metrics')
        def on_metrics(data):
            # Update metric histories
            if 'reward' in data:
                self.reward_history.append(data['reward'])
            if 'loss' in data:
                self.loss_history.append(data['loss'])
            if 'episode_profit' in data:
                self.profit_history.append(data['episode_profit'])
                
    def get_gpu_metrics(self):
        """Get detailed GPU metrics for training"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpus.append({
                            'index': int(parts[0]),
                            'util': float(parts[2]),
                            'memory_used': float(parts[3]),
                            'memory_total': float(parts[4]),
                            'temp': float(parts[5])
                        })
                return gpus
        except:
            pass
        return []
        
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio from profit history"""
        if len(self.profit_history) < 2:
            return 0.0
        
        returns = []
        for i in range(1, len(self.profit_history)):
            if self.profit_history[i-1] != 0:
                ret = (self.profit_history[i] - self.profit_history[i-1]) / abs(self.profit_history[i-1])
                returns.append(ret)
        
        if not returns:
            return 0.0
            
        avg_return = statistics.mean(returns)
        std_return = statistics.stdev(returns) if len(returns) > 1 else 1
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (avg_return * 252**0.5) / std_return if std_return > 0 else 0
        return sharpe
        
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown from profit history"""
        if len(self.profit_history) < 2:
            return 0.0
            
        peak = self.profit_history[0]
        max_dd = 0
        
        for value in self.profit_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
            
        return max_dd * 100  # Return as percentage
        
    def get_session_details(self):
        """Get detailed session information"""
        if not self.current_session:
            return None
            
        try:
            session_id = self.current_session.get('session_id')
            response = requests.get(f"{self.api_url}/api/sessions/{session_id}")
            if response.ok:
                return response.json()
        except:
            pass
        return self.current_session
        
    def display_progress_bar(self, current, total, width=50):
        """Display a progress bar"""
        if total == 0:
            return "N/A"
            
        progress = current / total
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        percentage = progress * 100
        
        return f"[{bar}] {percentage:.1f}%"
        
    def format_time_elapsed(self):
        """Format elapsed training time"""
        if not self.start_time:
            return "00:00:00"
            
        elapsed = datetime.now() - self.start_time
        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        seconds = elapsed.seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def monitor(self):
        """Start comprehensive training monitoring"""
        print("=" * 100)
        print("🤖 AI Trading System - Advanced Training Monitor")
        print("=" * 100)
        
        # Check health
        try:
            health = requests.get(f"{self.api_url}/health").json()
            print(f"✅ System Status: {health['status']}")
        except:
            print("❌ System is not responding")
            return
            
        # Connect WebSocket
        try:
            self.sio.connect(self.api_url)
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            
        print("\n📊 Monitoring Training Progress... Press Ctrl+C to stop")
        print("-" * 100)
        
        try:
            while True:
                # Clear screen for refresh
                print("\033[H\033[J", end='')
                
                # Header
                print(f"🤖 AI Trading System - Training Monitor [{datetime.now().strftime('%H:%M:%S')}]")
                print("=" * 100)
                
                # Get session details
                session = self.get_session_details()
                if session:
                    # Training Progress
                    current_ep = session.get('current_episode', 0)
                    total_ep = session.get('total_episodes', 1)
                    print(f"\n📈 TRAINING PROGRESS")
                    print(f"Session: {session.get('name', 'N/A')} | Algorithm: {session.get('algorithm_type', 'ANE-PPO')}")
                    print(f"Episode: {current_ep}/{total_ep}")
                    print(self.display_progress_bar(current_ep, total_ep, 60))
                    print(f"Time Elapsed: {self.format_time_elapsed()}")
                    
                    # Financial Metrics
                    print(f"\n💰 FINANCIAL PERFORMANCE")
                    total_profit = session.get('total_profit', 0)
                    profit_color = '\033[92m' if total_profit > 0 else '\033[91m'
                    print(f"Total P&L: {profit_color}${total_profit:,.2f}\033[0m")
                    
                    win_rate = session.get('win_rate', 0)
                    print(f"Win Rate: {win_rate:.1f}% ({self.win_count}W / {self.loss_count}L)")
                    print(f"Total Trades: {session.get('total_trades', 0)}")
                    
                    # Risk Metrics
                    sharpe = self.calculate_sharpe_ratio()
                    max_dd = self.calculate_max_drawdown()
                    print(f"\n📊 RISK METRICS")
                    print(f"Sharpe Ratio: {sharpe:.2f}")
                    print(f"Max Drawdown: {max_dd:.1f}%")
                    print(f"Risk Level: {session.get('parameters', {}).get('risk_level', 'moderate')}")
                    
                    # Training Metrics
                    print(f"\n🧠 LEARNING METRICS")
                    if self.reward_history:
                        avg_reward = statistics.mean(list(self.reward_history)[-20:])
                        print(f"Avg Reward (last 20): {avg_reward:.4f}")
                    if self.loss_history:
                        avg_loss = statistics.mean(list(self.loss_history)[-20:])
                        print(f"Avg Loss (last 20): {avg_loss:.4f}")
                    
                    # Algorithm Details
                    params = session.get('parameters', {})
                    print(f"\n⚙️  ALGORITHM CONFIGURATION")
                    print(f"Learning Rate: {params.get('learning_rate', 3e-4)}")
                    print(f"Batch Size: {params.get('batch_size', 64)}")
                    print(f"Transformer Layers: {params.get('transformer_layers', 6)}")
                    print(f"Attention Heads: {params.get('attention_heads', 8)}")
                    
                    # GPU Utilization
                    gpus = self.get_gpu_metrics()
                    if gpus:
                        print(f"\n🎮 GPU UTILIZATION")
                        for gpu in gpus:
                            util_bar = self.display_progress_bar(gpu['util'], 100, 30)
                            print(f"GPU {gpu['index']}: {util_bar} {gpu['util']:.1f}% | "
                                  f"Mem: {gpu['memory_used']:.0f}/{gpu['memory_total']:.0f}MB | "
                                  f"Temp: {gpu['temp']:.0f}°C")
                    
                    # Recent Performance
                    if len(self.profit_history) > 10:
                        print(f"\n📈 RECENT PERFORMANCE (Last 10 Episodes)")
                        recent_profits = list(self.profit_history)[-10:]
                        profit_trend = "📈" if recent_profits[-1] > recent_profits[0] else "📉"
                        print(f"Trend: {profit_trend}")
                        print("Episodes: " + " ".join([f"{'🟢' if p > 0 else '🔴'}" for p in recent_profits]))
                        
                else:
                    print("\n⏳ Waiting for training session to start...")
                    print("   Start a training session via the API to see metrics")
                    
                    # Still show GPU status
                    gpus = self.get_gpu_metrics()
                    if gpus:
                        print(f"\n🎮 GPU STATUS")
                        for gpu in gpus:
                            print(f"GPU {gpu['index']}: {gpu['util']:.1f}% utilization | "
                                  f"{gpu['memory_used']:.0f}MB used | {gpu['temp']:.0f}°C")
                
                print("\n" + "-" * 100)
                print("Press Ctrl+C to stop monitoring")
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            if self.sio.connected:
                self.sio.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Advanced Training Monitor for AI Trading System')
    parser.add_argument('--url', default='http://localhost:5000', 
                        help='API URL (default: http://localhost:5000)')
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.url)
    monitor.monitor()

if __name__ == '__main__':
    main()