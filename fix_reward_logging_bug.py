#!/usr/bin/env python3
"""
Fix the reward logging bug where portfolio values are being logged as rewards
"""

import os
import re

def fix_reward_bug():
    """
    The bug is that portfolio values (around 1214.79) are being logged as rewards.
    This happens when the agent stops trading and has 0 trades per episode.
    """
    
    print("Analyzing the reward bug...")
    print("="*60)
    
    # Check train_standalone.py
    with open('train_standalone.py', 'r') as f:
        content = f.read()
        
    # Find where episode_reward is being accumulated
    reward_accumulation = re.search(r'episode_reward \+= reward', content)
    if reward_accumulation:
        print("✓ Found correct reward accumulation: episode_reward += reward")
    else:
        print("✗ ERROR: Reward accumulation not found!")
        
    # Check if we're passing episode_reward to tracker.end_episode
    tracker_call = re.search(r'tracker\.end_episode\(episode_reward', content)
    if tracker_call:
        print("✓ Found tracker.end_episode(episode_reward, ...) call")
    else:
        print("✗ ERROR: tracker.end_episode call not found!")
        
    # Check futures_env_realistic.py
    print("\nChecking futures_env_realistic.py...")
    with open('futures_env_realistic.py', 'r') as f:
        env_content = f.read()
        
    # Check the step method return
    step_return = re.search(r'return obs, reward, self\.done', env_content)
    if step_return:
        print("✓ step() correctly returns (obs, reward, done, info)")
    else:
        print("✗ ERROR: step() return statement issue!")
        
    # Check get_reward method
    if 'def get_reward' in env_content:
        print("✓ get_reward() method exists")
        
        # Extract get_reward method
        start = env_content.find('def get_reward')
        end = env_content.find('\n    def ', start + 1)
        if end == -1:
            end = len(env_content)
        get_reward_code = env_content[start:end]
        
        # Check for portfolio value returns
        if 'self.cash' in get_reward_code or 'self.portfolio' in get_reward_code:
            print("✗ ERROR: get_reward() might be returning portfolio values!")
        else:
            print("✓ get_reward() doesn't directly return portfolio values")
            
    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("The rewards 1214.79, 1224.74 etc. are portfolio/cash values.")
    print("This happens when:")
    print("1. Agent stops trading (0 trades per episode)")
    print("2. Something is logging portfolio value as 'reward'")
    print("\nPossible causes:")
    print("1. Observation values being mistaken for rewards")
    print("2. Portfolio tracking code interfering with reward tracking")
    print("3. Debug logging that's incorrectly labeling values")
    
    # Create a patch to add more debugging
    patch_code = '''
# Add this to train_standalone.py right after line 526 (episode_reward += reward)
if abs(reward) > 100:
    loggers['algorithm'].error(
        f"LARGE STEP REWARD: {reward:.2f} at episode {episode}, step {step}"
    )
if episode > 60 and env.trades_this_episode == 0 and abs(episode_reward) > 1000:
    loggers['algorithm'].error(
        f"PORTFOLIO VALUE AS REWARD BUG: episode_reward={episode_reward:.2f}"
    )
'''
    
    print("\nSUGGESTED FIX:")
    print("Add the following debug code to train_standalone.py:")
    print(patch_code)
    
    # Check if there's any code that might be modifying episode_reward
    portfolio_pattern = re.findall(r'episode_reward\s*=\s*[^+]', content)
    if len(portfolio_pattern) > 1:  # More than just initialization
        print("\n⚠️ WARNING: Found multiple episode_reward assignments:")
        for match in portfolio_pattern:
            print(f"  {match}")
            
    return True

if __name__ == "__main__":
    fix_reward_bug()
    
    print("\n\nNEXT STEPS:")
    print("1. Run: python fix_database_tables.py")
    print("2. Check if train_standalone.py has been modified")
    print("3. Look for any custom logging that might be mislabeling values")
    print("4. The bug appears after episode 60 when agent stops trading")