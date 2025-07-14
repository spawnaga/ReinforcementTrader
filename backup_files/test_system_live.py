"""
Quick test to verify the trading system is working
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_system():
    print("Testing Trading System...")
    
    # 1. Test system status
    print("\n1. Testing system status...")
    response = requests.get(f"{BASE_URL}/api/system_test")
    print(f"System test response: {response.json()}")
    
    # 2. Check sessions
    print("\n2. Checking existing sessions...")
    response = requests.get(f"{BASE_URL}/api/sessions")
    sessions = response.json()
    print(f"Found {len(sessions)} sessions")
    
    # 3. Force start training
    print("\n3. Force starting training...")
    response = requests.post(f"{BASE_URL}/api/force_start_training")
    if response.status_code == 200:
        result = response.json()
        print(f"Training started: {result}")
        session_id = result.get('session_id')
        
        # 4. Wait a bit and check trades
        print("\n4. Waiting 5 seconds for trades...")
        time.sleep(5)
        
        # 5. Check recent trades
        print("\n5. Checking recent trades...")
        response = requests.get(f"{BASE_URL}/api/recent_trades?limit=10")
        trades = response.json()
        print(f"Found {len(trades)} trades")
        for trade in trades[:3]:
            print(f"  - {trade.get('position_type')} @ ${trade.get('entry_price', 0):.2f}")
    else:
        print(f"Failed to start training: {response.text}")

if __name__ == "__main__":
    test_system()