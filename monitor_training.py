#!/usr/bin/env python3
"""
Monitor training sessions in real-time with detailed trade information
"""
import requests
import time
import sys
from datetime import datetime
import socketio

# Initialize Socket.IO client for real-time updates
sio = socketio.Client()
latest_trades = []
latest_metrics = {}

@sio.event
def connect():
    print("✅ Connected to WebSocket")

@sio.event
def disconnect():
    print("❌ Disconnected from WebSocket")

@sio.event
def new_trade(data):
    """Handle new trade events"""
    global latest_trades
    latest_trades.append(data)
    # Keep only last 10 trades
    if len(latest_trades) > 10:
        latest_trades.pop(0)

@sio.event
def training_update(data):
    """Handle training update events"""
    global latest_metrics
    session_id = data.get('session_id')
    latest_metrics[session_id] = data

def monitor_sessions():
    """Monitor active training sessions with detailed information"""
    api_url = "http://127.0.0.1:5000"
    
    # Connect to WebSocket
    try:
        sio.connect(api_url)
    except Exception as e:
        print(f"⚠️  WebSocket connection failed: {e}")
    
    print("🔍 Enhanced Training Monitor Started...")
    print("=" * 80)
    
    last_episode = {}
    trade_count = {}
    
    while True:
        try:
            # Get all sessions
            response = requests.get(f"{api_url}/api/sessions")
            sessions = response.json()
            
            # Find active sessions
            active_sessions = [s for s in sessions if s['status'] == 'active']
            
            # Clear screen (works on Unix-like systems)
            print("\033[2J\033[H")
            
            print(f"📊 Enhanced Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            
            if not active_sessions:
                print("❌ No active training sessions found")
                print("\n💡 Start a new session with:")
                print('   curl -X POST http://127.0.0.1:5000/api/start_training \\')
                print('     -H "Content-Type: application/json" \\')
                print('     -d \'{"session_name": "My Session", "algorithm_type": "ANE_PPO", "total_episodes": 100}\'')
            else:
                for session in active_sessions:
                    session_id = session['id']
                    current_episode = session['current_episode']
                    
                    # Calculate episode progress rate
                    if session_id in last_episode:
                        episode_rate = current_episode - last_episode[session_id]
                    else:
                        episode_rate = 0
                    last_episode[session_id] = current_episode
                    
                    print(f"\n🚀 Session: {session['name']} (ID: {session_id})")
                    print(f"   Algorithm: {session['algorithm_type']}")
                    print(f"   Status: {session['status']}")
                    print(f"   Progress: {current_episode}/{session['total_episodes']} episodes")
                    
                    # Progress bar
                    progress_pct = (current_episode / session['total_episodes']) * 100 if session['total_episodes'] > 0 else 0
                    bar_length = 30
                    filled_length = int(bar_length * progress_pct // 100)
                    bar = '█' * filled_length + '░' * (bar_length - filled_length)
                    print(f"   [{bar}] {progress_pct:.1f}% (Rate: {episode_rate} eps/update)")
                    
                    print(f"\n   💰 Performance:")
                    print(f"      Total Trades: {session['total_trades']}")
                    print(f"      Total Profit: ${session['total_profit']:.2f}")
                    print(f"      Win Rate: {session['win_rate']*100:.1f}%")
                    print(f"      Sharpe Ratio: {session['sharpe_ratio']:.2f}")
                    print(f"      Max Drawdown: {session['max_drawdown']*100:.1f}%")
                    
                    # Get recent trades for this session
                    trades_response = requests.get(f"{api_url}/api/trades?session_id={session_id}&limit=5")
                    if trades_response.ok:
                        recent_trades = trades_response.json()
                        if recent_trades:
                            print(f"\n   📈 Recent Trades:")
                            for trade in recent_trades[:5]:
                                action = "BUY" if trade['position_type'] == 'long' else "SELL"
                                status_icon = "✅" if trade['profit_loss'] > 0 else "❌"
                                print(f"      {status_icon} {action} @ {trade['entry_price']:.2f} → "
                                      f"{trade['exit_price']:.2f if trade['exit_price'] else 'OPEN'} "
                                      f"(P/L: ${trade['profit_loss']:.2f})")
                    
                    # Show latest metrics from WebSocket
                    if session_id in latest_metrics:
                        metrics = latest_metrics[session_id]
                        print(f"\n   🧠 Latest Training Metrics:")
                        print(f"      Episode: {metrics.get('episode', 'N/A')}")
                        print(f"      Reward: {metrics.get('reward', 0):.2f}")
                        print(f"      Loss: {metrics.get('loss', 0):.4f}")
            
            # Check system resources
            try:
                perf_response = requests.get(f"{api_url}/api/status")
                if perf_response.ok:
                    status = perf_response.json()
                    print(f"\n💻 System Resources:")
                    print(f"   Active Sessions: {status['active_sessions']}")
                    print(f"   Total Sessions: {status['total_sessions']}")
                    print(f"   Total Trades: {status['total_trades']}")
            except:
                pass
            
            print("\n" + "=" * 80)
            print("Press Ctrl+C to stop monitoring | Updates every 2 seconds")
            
            # Wait 2 seconds before next update
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\n✅ Monitoring stopped")
            sio.disconnect()
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_sessions()