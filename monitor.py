#!/usr/bin/env python3
"""
Simple monitoring script for AI Trading System

Usage:
    python monitor.py [--url http://localhost:5000]
"""

import argparse
import requests
import socketio
import time
from datetime import datetime
import sys
import logging

logger = logging.getLogger(__name__)

class TradingMonitor:
    def __init__(self, api_url):
        self.api_url = api_url.rstrip('/')
        self.sio = socketio.Client()
        self.setup_handlers()
        
    def setup_handlers(self):
        """Set up WebSocket event handlers"""
        
        @self.sio.on('connect')
        def on_connect():
            print(f"✅ Connected to {self.api_url}")
            
        @self.sio.on('performance_metrics')
        def on_performance(data):
            print(f"\n📊 System Performance [{datetime.now().strftime('%H:%M:%S')}]")
            print(f"   CPU: {data.get('cpu_usage', 0):.1f}%")
            print(f"   Memory: {data.get('memory_usage', 0):.1f}%")
            print(f"   GPU (avg): {data.get('gpu_usage', 0):.1f}%")
            print(f"   Network: {data.get('network_io', 0):.2f} MB/s")
            
        @self.sio.on('trade_update')
        def on_trade(data):
            print(f"\n💰 New Trade:")
            print(f"   Type: {data.get('position_type', 'N/A')}")
            print(f"   Entry: ${data.get('entry_price', 0):.2f}")
            if data.get('exit_price'):
                print(f"   Exit: ${data.get('exit_price', 0):.2f}")
                print(f"   P&L: ${data.get('profit_loss', 0):.2f}")
                
        @self.sio.on('session_update')
        def on_session(data):
            print(f"\n🎯 Training Update:")
            print(f"   Episode: {data.get('episode', 0)}/{data.get('total_episodes', 0)}")
            print(f"   Reward: {data.get('reward', 0):.2f}")
            
    def get_system_status(self):
        """Get current system status via API"""
        try:
            response = requests.get(f"{self.api_url}/api/status")
            if response.ok:
                return response.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Failed to get system status: {e}")
        return None
        
    def get_active_sessions(self):
        """Get active training sessions"""
        try:
            response = requests.get(f"{self.api_url}/api/sessions?status=running")
            if response.ok:
                return response.json()
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"Failed to get active sessions: {e}")
        return []
        
    def get_gpu_info(self):
        """Get detailed GPU information"""
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory', 
                                   '--format=csv,noheader,nounits'], capture_output=True, text=True)
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        gpus.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'memory_used': int(parts[2]),
                            'memory_total': int(parts[3]),
                            'gpu_util': float(parts[4]),
                            'memory_util': float(parts[5])
                        })
                return gpus
        except (subprocess.CalledProcessError, ValueError, ImportError) as e:
            logger.warning(f"Failed to get GPU info: {e}")
        return []
    
    def monitor(self):
        """Start monitoring the system"""
        print("=" * 60)
        print("AI Trading System Monitor")
        print("=" * 60)
        
        # Check health
        try:
            health = requests.get(f"{self.api_url}/health").json()
            print(f"✅ System Status: {health['status']}")
        except (requests.RequestException, ValueError) as e:
            print(f"❌ System is not responding: {e}")
            return
            
        # Show GPU information
        gpus = self.get_gpu_info()
        if gpus:
            print(f"\n🎮 GPU Configuration ({len(gpus)} devices):")
            for gpu in gpus:
                print(f"   GPU {gpu['index']}: {gpu['name']}")
                print(f"      Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_util']:.1f}%)")
                print(f"      Utilization: {gpu['gpu_util']:.1f}%")
            
        # Get initial status
        status = self.get_system_status()
        if status:
            print(f"\n📈 System Overview:")
            print(f"   Active Sessions: {status.get('active_sessions', 0)}")
            print(f"   Total Sessions: {status.get('total_sessions', 0)}")
            print(f"   Total Trades: {status.get('total_trades', 0)}")
            
        # Connect WebSocket
        print(f"\n🔌 Connecting to WebSocket...")
        try:
            self.sio.connect(self.api_url)
        except Exception as e:
            print(f"❌ WebSocket connection failed: {e}")
            return
            
        print("\n📡 Monitoring active... Press Ctrl+C to stop")
        print("-" * 60)
        
        try:
            # Keep monitoring
            while True:
                # Periodically check sessions
                sessions = self.get_active_sessions()
                if sessions:
                    print(f"\n🏃 Active Training Sessions: {len(sessions)}")
                    for session in sessions[:3]:  # Show max 3
                        print(f"   - {session.get('session_name', 'Unknown')}: Episode {session.get('current_episode', 0)}")
                
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            self.sio.disconnect()

def main():
    parser = argparse.ArgumentParser(description='Monitor AI Trading System')
    parser.add_argument('--url', default='http://localhost:5000', 
                        help='API URL (default: http://localhost:5000)')
    args = parser.parse_args()
    
    monitor = TradingMonitor(args.url)
    monitor.monitor()

if __name__ == '__main__':
    main()