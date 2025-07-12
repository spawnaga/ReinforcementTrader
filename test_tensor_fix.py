#!/usr/bin/env python3
"""Test script to verify the tensor dimension fix for transformer_attention.py"""

import torch
import sys
import numpy as np
from rl_algorithms.q_learning import DQN

def test_tensor_dimension_fix():
    """Test if the tensor dimension error is fixed"""
    print("Testing tensor dimension fix...")
    
    # Create a test DQN model
    input_dim = 60  # Number of features from trading environment
    output_dim = 3  # Number of actions (buy, sell, hold)
    hidden_dim = 512
    
    try:
        # Initialize DQN
        model = DQN(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim)
        print(f"✓ DQN model created successfully with input_dim={input_dim}, hidden_dim={hidden_dim}")
        
        # Create test input - batch of sequences
        batch_size = 2
        seq_len = 100
        test_input = torch.randn(batch_size, seq_len, input_dim)
        print(f"✓ Test input shape: {test_input.shape}")
        
        # Forward pass
        output = model(test_input)
        print(f"✓ Forward pass successful! Output shape: {output.shape}")
        
        # Test with single sequence
        single_input = torch.randn(1, seq_len, input_dim)
        single_output = model(single_input)
        print(f"✓ Single sequence test successful! Output shape: {single_output.shape}")
        
        print("\n🎉 SUCCESS: Tensor dimension error has been fixed!")
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tensor_dimension_fix()
    sys.exit(0 if success else 1)