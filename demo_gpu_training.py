#!/usr/bin/env python
"""
GPU and Training Parameters Demo
Shows all the available options for controlling GPUs and training loops
"""

print("=== AI Trading System - GPU & Training Control ===\n")

print("✅ YES - The system includes full control over:\n")

print("1️⃣ GPU CONFIGURATION:")
print("   --device cpu/gpu/auto    → Choose CPU, GPU, or auto-detect")
print("   --gpu-ids 0 1 2 3       → Use specific GPUs (example: 4 GPUs)")
print("   --num-gpus 2            → Use first 2 available GPUs")
print("")

print("2️⃣ TRAINING EPISODES:")
print("   --episodes 1000         → Number of episodes (default: 1000)")
print("   --episodes 5000         → Train for 5,000 episodes")
print("   --episodes 10000        → Train for 10,000 episodes")
print("")

print("3️⃣ TRAINING LOOPS & STEPS:")
print("   --max-steps 200         → Steps per episode (default: 200)")
print("   --training-loops 5      → Number of loops per episode")
print("   --epochs-per-loop 10    → Epochs in each training loop")
print("   --batch-size 64         → Batch size for training")
print("")

print("=== COMPLETE EXAMPLES ===\n")

examples = [
    ("Use 4 GPUs for 5000 episodes:",
     "python trading_cli.py --device gpu --num-gpus 4 --episodes 5000 --ticker NQ"),
    
    ("Use specific GPUs (0,1,2) with 10,000 episodes:",
     "python trading_cli.py --gpu-ids 0 1 2 --episodes 10000 --ticker ES"),
    
    ("CPU training with custom loops:",
     "python trading_cli.py --device cpu --episodes 1000 --training-loops 5 --epochs-per-loop 10"),
    
    ("Full GPU training with all parameters:",
     "python trading_cli.py --device gpu --num-gpus 4 --episodes 5000 --max-steps 300 --training-loops 3 --epochs-per-loop 5 --batch-size 128 --ticker NQ --algorithm ane-ppo"),
    
    ("Multi-GPU with risk management:",
     "python trading_cli.py --gpu-ids 0 1 2 3 --episodes 10000 --max-trades 10 --stop-loss 50 --take-profit 100")
]

for desc, cmd in examples:
    print(f"{desc}")
    print(f"  $ {cmd}\n")

print("=== HARDWARE DETECTION ===")
print("The system automatically detects:")
print("• Number of available GPUs")
print("• GPU memory and capabilities")
print("• Optimal batch sizes for your hardware")
print("• Multi-GPU training with PyTorch DataParallel\n")

print("=== CURRENT SYSTEM STATUS ===")
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPUs detected: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("✗ Running in CPU mode (no GPUs detected)")

print("\nReady to train with your preferred configuration!")