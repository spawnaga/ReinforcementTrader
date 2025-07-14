#!/usr/bin/env python3
"""
Start training with more aggressive exploration settings
"""
import requests
import json

def start_aggressive_training():
    """Start new training with exploration-friendly settings"""
    
    # First, stop any active sessions
    print("🛑 Stopping active sessions...")
    sessions_response = requests.get("http://127.0.0.1:5000/api/sessions")
    if sessions_response.status_code == 200:
        sessions = sessions_response.json()
        for session in sessions:
            if session.get('status') == 'active':
                stop_url = f"http://127.0.0.1:5000/api/sessions/{session['id']}/stop"
                requests.post(stop_url)
                print(f"   Stopped session {session['id']}")
    
    # Start new training with custom parameters
    print("\n🚀 Starting aggressive training session...")
    
    config = {
        "session_name": "Aggressive Explorer",
        "algorithm_type": "ANE_PPO",
        "total_episodes": 500,
        "learning_rate": 0.0005,  # Higher learning rate
        "clip_range": 0.3,        # Wider clip range for more exploration
        "entropy_coef": 0.05,     # Higher entropy for more random actions
        "n_steps": 512,           # Smaller steps for faster updates
        "batch_size": 32          # Smaller batches
    }
    
    response = requests.post(
        "http://127.0.0.1:5000/api/start_training",
        headers={"Content-Type": "application/json"},
        data=json.dumps(config)
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Training started!")
        print(f"   Session ID: {result.get('session_id')}")
        print(f"   Session Name: {config['session_name']}")
        print(f"\n📊 Monitor with: python monitor_training.py")
        print(f"\n💡 This session will:")
        print(f"   - Take more random actions (higher entropy)")
        print(f"   - Learn faster from experiences")
        print(f"   - Update policy more frequently")
    else:
        print(f"❌ Failed to start training: {response.text}")

if __name__ == "__main__":
    start_aggressive_training()