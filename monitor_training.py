#!/usr/bin/env python3
"""
Monitor training sessions in real-time
"""
import requests
import time
import sys
from datetime import datetime

def monitor_sessions():
    """Monitor active training sessions"""
    api_url = "http://127.0.0.1:5000"
    
    print("🔍 Monitoring Training Sessions...")
    print("=" * 60)
    
    while True:
        try:
            # Get all sessions
            response = requests.get(f"{api_url}/api/sessions")
            sessions = response.json()
            
            # Find active sessions
            active_sessions = [s for s in sessions if s['status'] == 'active']
            
            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H")
            
            print(f"📊 Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 60)
            
            if not active_sessions:
                print("❌ No active training sessions")
            else:
                for session in active_sessions:
                    print(f"\n🚀 Session: {session['name']} (ID: {session['id']})")
                    print(f"   Algorithm: {session['algorithm_type']}")
                    print(f"   Progress: {session['current_episode']}/{session['total_episodes']} episodes")
                    print(f"   Total Trades: {session['total_trades']}")
                    print(f"   Total Profit: ${session['total_profit']:.2f}")
                    print(f"   Win Rate: {session['win_rate']*100:.1f}%")
                    print(f"   Sharpe Ratio: {session['sharpe_ratio']:.2f}")
                    print(f"   Max Drawdown: {session['max_drawdown']*100:.1f}%")
            
            # Also check system status
            status_response = requests.get(f"{api_url}/api/status")
            if status_response.ok:
                status = status_response.json()
                print(f"\n📈 System Status: {status['system_status']}")
                print(f"   Active Sessions: {status['active_sessions']}")
                print(f"   Total Sessions: {status['total_sessions']}")
                print(f"   Total Trades: {status['total_trades']}")
            
            print("\n" + "=" * 60)
            print("Press Ctrl+C to stop monitoring")
            
            # Wait 2 seconds before next update
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_sessions()