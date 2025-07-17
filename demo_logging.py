#!/usr/bin/env python3
"""
Demo script to showcase the professional logging system
Run this to see how the logging works
"""
import time
from datetime import datetime
from logging_config import get_loggers
from tqdm import tqdm

def demo_logging():
    """Demonstrate the professional logging system"""
    print("=== Professional Logging System Demo ===\n")
    
    # Initialize logging
    loggers = get_loggers()
    session_timestamp = loggers['session_timestamp']
    
    print(f"✓ Logs created in: logs/{session_timestamp}/")
    print("  • trading.log    - Trade executions")
    print("  • positions.log  - Position tracking")
    print("  • rewards.log    - Reward calculations")
    print("  • algorithm.log  - Algorithm decisions")
    print("  • performance.log - Learning metrics")
    print("\nWatch the tqdm progress bars below:\n")
    
    # Simulate training with tqdm
    episodes = 10
    episode_pbar = tqdm(range(episodes), desc="Training Episodes", unit="ep")
    
    for episode in episode_pbar:
        # Log to algorithm.log
        loggers['algorithm'].info(f"Starting Episode {episode + 1}")
        
        # Simulate steps with nested progress bar
        steps = 50
        step_pbar = tqdm(range(steps), desc=f"Episode {episode+1}", unit="steps", leave=False)
        
        total_reward = 0
        for step in step_pbar:
            # Simulate trading decision
            if step % 10 == 0:
                # Log trade
                loggers['trading'].info(
                    f"Episode {episode+1} | Step {step} | "
                    f"LONG Entry @ $3500.25 | Target: $3570.25"
                )
                
                # Log position
                loggers['positions'].info(
                    f"12:34:56 | Position: +1 | Price: $3500.25 | "
                    f"Unrealized: $0.00 | Account: $100,000.00"
                )
            
            # Simulate reward
            reward = 10.5 if step % 5 == 0 else -2.5
            total_reward += reward
            
            if step % 15 == 0:
                # Log reward
                loggers['rewards'].info(
                    f"Episode {episode+1} Trade {step//15}: "
                    f"${reward:.2f} | Cumulative: ${total_reward:.2f}"
                )
            
            # Update progress bar
            step_pbar.set_postfix({'reward': f'{total_reward:.2f}'})
            time.sleep(0.01)  # Small delay for demo
        
        step_pbar.close()
        
        # Log episode performance
        loggers['performance'].info(
            f"Episode {episode+1} | Reward: {total_reward:.2f} | "
            f"Profit: ${total_reward*5:.2f} | Trades: 3 | "
            f"Win Rate: 66.7% | Sharpe: 1.25"
        )
        
        # Update main progress bar
        episode_pbar.set_postfix({'avg_reward': f'{total_reward:.2f}'})
    
    # Final summary
    loggers['performance'].info("="*60)
    loggers['performance'].info("TRAINING COMPLETED!")
    loggers['performance'].info("✓ AGENT LEARNED! Reward improved by 125%")
    loggers['performance'].info("="*60)
    
    print(f"\n✓ Demo completed! Check logs in: logs/{session_timestamp}/")
    print("  You can also check: logs/latest/ (symlink)")

if __name__ == "__main__":
    demo_logging()