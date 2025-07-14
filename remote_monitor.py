#!/usr/bin/env python3
"""
Remote Trading Monitor - Connect to your trading system from any machine
"""
import requests
import time
import os
import sys
from datetime import datetime
import json

class RemoteMonitor:
    def __init__(self, host="localhost", port=5000):
        self.base_url = f"http://{host}:{port}"
        self.session = requests.Session()
        
    def clear_screen(self):
        """Clear screen cross-platform"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def check_connection(self):
        """Check if we can connect to the trading system"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_active_sessions(self):
        """Get active training sessions"""
        try:
            response = self.session.get(f"{self.base_url}/api/sessions")
            if response.status_code == 200:
                sessions = response.json()
                return [s for s in sessions if s['status'] == 'active']
            return []
        except:
            return []
    
    def get_session_details(self, session_id):
        """Get detailed session information"""
        try:
            response = self.session.get(f"{self.base_url}/api/sessions/{session_id}")
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def get_recent_trades(self, session_id, limit=5):
        """Get recent trades for a session"""
        try:
            response = self.session.get(f"{self.base_url}/api/recent_trades?session_id={session_id}&limit={limit}")
            if response.status_code == 200:
                return response.json()
            return []
        except:
            return []
    
    def display_monitor(self):
        """Display the monitoring interface"""
        self.clear_screen()
        
        print("=" * 80)
        print(f"{'🚀 Remote Trading Monitor':^80}")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^80}")
        print("=" * 80)
        
        # Check connection
        if not self.check_connection():
            print("\n❌ Cannot connect to trading system!")
            print(f"   Make sure the system is running at {self.base_url}")
            print("\n   To connect to a remote machine:")
            print(f"   python remote_monitor.py <HOST_IP> <PORT>")
            return
        
        # Get active sessions
        active_sessions = self.get_active_sessions()
        
        if not active_sessions:
            print("\n❌ No active training sessions")
            print("\n   Start training on the remote machine first")
            return
        
        # Display each active session
        for session in active_sessions:
            session_details = self.get_session_details(session['id'])
            if not session_details:
                continue
                
            print(f"\n📊 Session: {session.get('name', 'Unknown')} (ID: {session['id']})")
            print(f"   Status: {session['status']}")
            print(f"   Started: {session.get('start_time', 'Unknown')}")
            
            # Algorithm info
            print(f"\n🧠 Algorithm: {session_details.get('algorithm', 'Unknown')}")
            if session_details.get('algorithm') == 'ANE-PPO':
                print("   • Transformer attention mechanisms")
                print("   • Multi-scale feature extraction")
                print("   • Actor-Critic architecture")
                print("   • Genetic algorithm optimization")
            
            # Training progress
            total_episodes = session_details.get('total_episodes', 10000)
            current_episode = session_details.get('current_episode', 0)
            progress = (current_episode / total_episodes) * 100
            
            print(f"\n🎯 Training Progress:")
            print(f"   Episodes: {current_episode}/{total_episodes}")
            
            # Progress bar
            bar_length = 50
            filled = int(bar_length * progress / 100)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"   [{bar}] {progress:.1f}%")
            
            # Performance metrics
            metrics = session_details.get('metrics', {})
            print(f"\n💰 Performance Metrics:")
            print(f"   Total Trades: {metrics.get('total_trades', 0)}")
            print(f"   Total Profit: ${metrics.get('total_profit', 0):,.2f}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1f}%")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.1f}%")
            
            # Recent trades
            recent_trades = self.get_recent_trades(session['id'])
            if recent_trades:
                print(f"\n📊 Recent Trades:")
                for trade in recent_trades[:5]:
                    if trade.get('exit_price'):
                        print(f"   #{trade['id']}: ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} P&L: ${trade.get('profit_loss', 0):.2f}")
                    else:
                        print(f"   #{trade['id']}: OPEN @ ${trade['entry_price']:.2f}")
        
        print("\n" + "=" * 80)
        print("Press Ctrl+C to quit | Updates every 5 seconds")
    
    def run(self):
        """Run the monitor loop"""
        try:
            while True:
                self.display_monitor()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")

def main():
    """Main entry point"""
    # Get host and port from command line
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    print(f"Connecting to trading system at {host}:{port}...")
    
    monitor = RemoteMonitor(host, port)
    
    # Test connection first
    print("Testing connection...")
    if monitor.check_connection():
        print("✓ Connection successful!")
        monitor.run()
    else:
        print("\n❌ Connection failed!")
        print(f"\nCannot connect to {host}:{port}")
        print("\nPossible issues:")
        print("1. The trading system is not running on the remote machine")
        print("2. The port 5000 is blocked by firewall")
        print("3. The Flask app is bound to localhost only (not 0.0.0.0)")
        print("\nTo fix on the remote machine:")
        print("1. Check if app is running: ps aux | grep python")
        print("2. Allow port: sudo ufw allow 5000")
        print("3. Edit run_local.py to use host='0.0.0.0'")
        print("\nOr use SSH tunnel:")
        print(f"ssh -L 5000:localhost:5000 username@{host}")
        print("Then run: python remote_monitor.py localhost 5000")

if __name__ == "__main__":
    main()