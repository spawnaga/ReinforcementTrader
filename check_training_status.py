#!/usr/bin/env python3
"""
Quick script to check training status and start a session if needed
"""
import requests
import json
import sys

API_URL = "http://localhost:5000"

def check_sessions():
    """Check active sessions"""
    try:
        response = requests.get(f"{API_URL}/api/sessions")
        sessions = response.json()
        print(f"Active sessions: {len(sessions)}")
        for session in sessions:
            print(f"  Session {session['id']}: {session['session_name']} - Status: {session['status']}")
            print(f"    Episodes: {session['current_episode']}/{session['total_episodes']}")
        return sessions
    except Exception as e:
        print(f"Error checking sessions: {e}")
        return []

def start_training():
    """Start a new training session"""
    payload = {
        "session_name": "NQ_Training_GPU_Test",
        "algorithm_type": "ANE_PPO",
        "total_episodes": 1000,
        "parameters": {
            "learning_rate": 0.0003,
            "batch_size": 64,
            "transformer_layers": 6,
            "attention_dim": 256
        }
    }
    
    try:
        response = requests.post(f"{API_URL}/api/start_training", json=payload)
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Training started successfully!")
            print(f"Session ID: {result['session_id']}")
            print(f"Algorithm: {result['algorithm']}")
            return result['session_id']
        else:
            print(f"\n❌ Failed to start training: {response.text}")
            return None
    except Exception as e:
        print(f"\n❌ Error starting training: {e}")
        return None

def main():
    print("🔍 Checking AI Trading System Status...")
    print("=" * 50)
    
    # Check existing sessions
    sessions = check_sessions()
    
    if not sessions:
        print("\n📊 No active sessions found.")
        choice = input("\nStart a new training session? (y/n): ")
        if choice.lower() == 'y':
            session_id = start_training()
            if session_id:
                print(f"\n🚀 Training session {session_id} is now running!")
                print("Check your dashboard at: http://localhost:5000/training_dashboard")
                print("Or monitor in terminal with: python training_monitor.py")
    else:
        print("\n✅ Training already in progress!")
        print("Check your dashboard at: http://localhost:5000/training_dashboard")

if __name__ == "__main__":
    main()