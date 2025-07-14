#!/usr/bin/env python3
"""
Quick test of the training system with reduced data
"""
import requests
import json
import time

def test_training():
    """Test the training system through the API"""
    base_url = "http://localhost:5000"
    
    print("Starting training test...")
    
    # Start a training session
    training_data = {
        "session_name": "quick_test",
        "algorithm_type": "ANE_PPO",
        "total_episodes": 10,  # Very small number for testing
        "parameters": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "clip_range": 0.2,
            "entropy_coef": 0.01
        }
    }
    
    try:
        response = requests.post(f"{base_url}/api/start_training", json=training_data)
        result = response.json()
        
        if result.get("success"):
            session_id = result.get("session_id")
            print(f"✓ Training started successfully! Session ID: {session_id}")
            
            # Wait a moment and check status
            time.sleep(5)
            
            # Get session status
            status_response = requests.get(f"{base_url}/api/session_status/{session_id}")
            status = status_response.json()
            print(f"Session status: {json.dumps(status, indent=2)}")
            
            # Wait for training to progress
            print("\nWaiting for training to progress...")
            for i in range(6):
                time.sleep(5)
                status_response = requests.get(f"{base_url}/api/session_status/{session_id}")
                status = status_response.json()
                if status.get("data"):
                    print(f"Episode: {status['data'].get('current_episode', 0)}/{status['data'].get('total_episodes', 0)}")
                    print(f"Status: {status['data'].get('status', 'unknown')}")
                    if status['data'].get('status') == 'completed':
                        print("✓ Training completed successfully!")
                        break
            
        else:
            print(f"✗ Failed to start training: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"✗ Error: {str(e)}")

if __name__ == "__main__":
    test_training()