#!/usr/bin/env python3
"""
Check the status of the remote trading system via SSH tunnel
"""
import requests
import json
import sys

def check_remote_system(host="localhost", port=5000):
    base_url = f"http://{host}:{port}"
    
    print(f"\n🔍 Checking remote trading system at {base_url}")
    print("=" * 60)
    
    # 1. Check if we can connect
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ Cannot connect to remote system at {base_url}")
            print("   Make sure your SSH tunnel is active:")
            print("   ssh -L 5000:localhost:5000 user@remote-host")
            return
        print("✅ Connected to remote trading system")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n   To connect to remote system:")
        print("   1. SSH tunnel: ssh -L 5000:localhost:5000 user@remote-host")
        print("   2. Then run: python check_remote_status.py")
        return
    
    # 2. Get all sessions
    try:
        response = requests.get(f"{base_url}/api/sessions")
        sessions = response.json() if response.status_code == 200 else []
        
        print(f"\n📊 Total sessions: {len(sessions)}")
        
        active_sessions = [s for s in sessions if s.get('status') == 'active']
        print(f"🟢 Active sessions: {len(active_sessions)}")
        
        if active_sessions:
            print("\n🎯 Active Training Sessions:")
            for session in active_sessions:
                print(f"\n   Session ID: {session.get('id')}")
                print(f"   Name: {session.get('name', session.get('session_name', 'Unknown'))}")
                print(f"   Algorithm: {session.get('algorithm_type')}")
                print(f"   Progress: {session.get('current_episode', 0)}/{session.get('total_episodes', 0)}")
                print(f"   Total Trades: {session.get('total_trades', 0)}")
                print(f"   Status: {session.get('status')}")
        else:
            print("\n⚠️  No active sessions found")
            print("\nAll sessions in database:")
            for session in sessions[-5:]:  # Show last 5 sessions
                print(f"   ID: {session.get('id')}, Status: {session.get('status')}, Episodes: {session.get('current_episode')}")
                
    except Exception as e:
        print(f"❌ Error getting sessions: {e}")
    
    # 3. Check recent trades
    try:
        response = requests.get(f"{base_url}/api/recent_trades?limit=10")
        if response.status_code == 200:
            trades = response.json()
            if trades:
                print(f"\n💰 Recent trades found: {len(trades)}")
                session_ids = list(set(t.get('session_id') for t in trades))
                print(f"   From sessions: {session_ids}")
            else:
                print("\n📈 No trades recorded yet")
    except Exception as e:
        print(f"❌ Error getting trades: {e}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # Default to localhost:5000 (for SSH tunnel)
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    check_remote_system(host, port)