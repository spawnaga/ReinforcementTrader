#!/usr/bin/env python3
"""
Debug script to check what sessions are available on the local server
"""
import requests
import json

def check_local_server(host="localhost", port=5000):
    base_url = f"http://{host}:{port}"
    
    print(f"Checking server at {base_url}")
    
    # Check health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print("✓ Server is healthy")
        else:
            print("✗ Server health check failed")
            return
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return
    
    # Get all sessions
    try:
        response = requests.get(f"{base_url}/api/sessions", timeout=5)
        print(f"\nGetting sessions: {response.status_code}")
        
        if response.status_code == 200:
            sessions = response.json()
            print(f"Total sessions found: {len(sessions)}")
            
            # Show all sessions
            for session in sessions:
                print(f"\n--- Session {session.get('id', 'Unknown')} ---")
                print(f"Name: {session.get('name', session.get('session_name', 'Unknown'))}")
                print(f"Status: {session.get('status', 'Unknown')}")
                print(f"Algorithm: {session.get('algorithm_type', 'Unknown')}")
                print(f"Current Episode: {session.get('current_episode', 0)}")
                print(f"Total Episodes: {session.get('total_episodes', 0)}")
                print(f"Total Trades: {session.get('total_trades', 0)}")
                
                # Check if this is an active session
                if session.get('status') == 'active':
                    print(">>> This is an ACTIVE session! <<<")
        else:
            print(f"Failed to get sessions: {response.text}")
            
    except Exception as e:
        print(f"Error getting sessions: {e}")
    
    # Get recent trades
    try:
        response = requests.get(f"{base_url}/api/recent_trades?limit=5", timeout=5)
        print(f"\n\nRecent trades check: {response.status_code}")
        
        if response.status_code == 200:
            trades = response.json()
            print(f"Recent trades found: {len(trades)}")
            if trades:
                print("Latest trade session IDs:", [t.get('session_id') for t in trades[:5]])
        else:
            print(f"Failed to get trades: {response.text}")
            
    except Exception as e:
        print(f"Error getting trades: {e}")

if __name__ == "__main__":
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    check_local_server(host, port)