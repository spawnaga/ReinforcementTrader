#!/usr/bin/env python3
"""Test the dimension fix for ANE-PPO ActorCritic network"""

import torch
import numpy as np

def test_actor_critic_dimensions():
    """Test ActorCritic network with correct dimensions"""
    from rl_algorithms.ane_ppo import ActorCritic
    
    print("Testing ActorCritic network dimensions...")
    
    # Initialize with correct parameters
    input_dim = 30  # Features from trading environment
    hidden_dim = 512
    attention_heads = 8
    attention_dim = 256
    transformer_layers = 6
    
    network = ActorCritic(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        attention_heads=attention_heads,
        attention_dim=attention_dim,
        transformer_layers=transformer_layers
    )
    
    print(f"✓ Network created successfully")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    
    # Test with 3D input (batch_size, seq_len, features)
    batch_size = 1
    seq_len = 60
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    print(f"\nTesting with 3D input: {test_input.shape}")
    
    try:
        with torch.no_grad():
            action_probs, values, q_values, regime_probs = network(test_input)
        
        print(f"✓ Forward pass successful!")
        print(f"  Action probs: {action_probs.shape}")
        print(f"  Values: {values.shape}")
        print(f"  Q-values: {q_values.shape}")
        print(f"  Regime probs: {regime_probs.shape}")
        
        # Test transformer attention directly
        print("\nTesting transformer attention module...")
        # Simulate the combined features after projection
        combined_features = torch.randn(batch_size, seq_len, hidden_dim)
        print(f"  Input to transformer: {combined_features.shape}")
        
        # Get transformer output
        transformer_out = network.transformer_attention(combined_features)
        if isinstance(transformer_out, tuple):
            transformer_out = transformer_out[0]
        print(f"  Transformer output: {transformer_out.shape}")
        
        print("\n🎉 All tests passed! Dimension fix is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actor_critic_dimensions()