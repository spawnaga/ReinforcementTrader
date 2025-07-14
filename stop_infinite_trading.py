#!/usr/bin/env python3
"""
Stop the infinite trading loop immediately
"""

import requests
import json
import time

def stop_infinite_trading():
    """Stop any sessions that are in an infinite trading loop"""
    
    print("Checking for infinite trading sessions...")
    
    # Get all active sessions
    try:
        response = requests.get('http://localhost:5000/api/sessions')
        sessions = response.json()
        
        # Find sessions with too many trades
        for session in sessions:
            if session.get('status') == 'active':
                session_id = session['id']
                print(f"\nChecking session {session_id}...")
                
                # Get recent trades for this session
                trades_response = requests.get(f'http://localhost:5000/api/trades?session_id={session_id}')
                if trades_response.status_code == 200:
                    trades = trades_response.json()
                    trade_count = len(trades)
                    
                    if trade_count > 100:
                        print(f"⚠️  Session {session_id} has {trade_count} trades - this is suspicious!")
                        
                        # Stop the session
                        print(f"Stopping session {session_id}...")
                        stop_response = requests.post(f'http://localhost:5000/api/stop_training')
                        if stop_response.status_code == 200:
                            print(f"✓ Successfully stopped session {session_id}")
                        else:
                            print(f"✗ Failed to stop session {session_id}: {stop_response.text}")
                    else:
                        print(f"✓ Session {session_id} has {trade_count} trades (normal)")
        
        print("\n✅ Infinite trading check complete!")
        print("\nTo prevent this in the future:")
        print("1. The environment has been patched with anti-exploitation measures")
        print("2. Trades at the same state are now blocked")
        print("3. Minimum 5-step gap enforced between trades")
        print("4. Maximum 5 trades per episode limit")
        
    except Exception as e:
        print(f"Error checking sessions: {str(e)}")

if __name__ == "__main__":
    stop_infinite_trading()