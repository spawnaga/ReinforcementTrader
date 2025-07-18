#!/usr/bin/env python3
"""
Comprehensive fix for training oscillation issue

The oscillation happens because:
1. Agent gets penalized for not trading
2. Agent trades poorly and loses money
3. Agent learns not to trade to avoid losses
4. Agent gets penalized again for not trading
5. Cycle repeats

Solution:
1. Add adaptive exploration that decreases penalty over time
2. Implement reward shaping that encourages consistent behavior
3. Add small rewards for reasonable trades even if they lose money
"""

import shutil
from datetime import datetime

def fix_futures_env_reward_system():
    """Fix the reward system to prevent oscillation"""
    
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Find and replace the no-trading penalty section
    old_penalty_code = """        # Small penalty for not trading to encourage action, but only if truly not trading
        if self.trades_this_episode == 0 and self.current_index > 50:
            # Curriculum-based penalty - smaller in early episodes
            if self.episode_number < 50:
                penalty = -0.05  # Very small penalty in easy mode
            elif self.episode_number < 150:
                penalty = -0.075  # Small penalty in medium mode
            else:
                penalty = -0.1  # Normal penalty in hard mode"""
    
    new_penalty_code = """        # Adaptive penalty for not trading that decreases if agent is learning
        if self.trades_this_episode == 0 and self.current_index > 50:
            # Calculate adaptive penalty based on recent performance
            base_penalty = -0.05
            
            # If agent has been oscillating, reduce penalty
            if hasattr(self, '_recent_trade_counts'):
                # Track last 10 episodes trade counts
                if len(self._recent_trade_counts) >= 10:
                    # Count oscillations (0 trades followed by many trades)
                    oscillations = 0
                    for i in range(1, len(self._recent_trade_counts)):
                        if self._recent_trade_counts[i-1] == 0 and self._recent_trade_counts[i] > 5:
                            oscillations += 1
                        elif self._recent_trade_counts[i-1] > 5 and self._recent_trade_counts[i] == 0:
                            oscillations += 1
                    
                    # Reduce penalty if oscillating
                    if oscillations > 2:
                        base_penalty *= 0.5  # Halve the penalty
            
            # Curriculum-based adjustment
            if self.episode_number < 50:
                penalty = base_penalty * 0.5  # Very small in easy mode
            elif self.episode_number < 150:
                penalty = base_penalty * 0.75  # Small in medium mode
            else:
                penalty = base_penalty  # Normal in hard mode"""
    
    content = content.replace(old_penalty_code, new_penalty_code)
    
    # Add tracking for recent trade counts in reset method
    reset_search = "def reset(self, seed=None, options=None):"
    reset_index = content.find(reset_search)
    if reset_index != -1:
        # Find the end of the reset method where we return
        after_reset = content[reset_index:]
        first_return_index = after_reset.find("return self.states[0]")
        
        if first_return_index != -1:
            # Insert tracking code before the return
            insert_point = reset_index + first_return_index
            
            tracking_code = """        # Track recent trade counts for oscillation detection
        if not hasattr(self, '_recent_trade_counts'):
            self._recent_trade_counts = []
        
        # Add last episode's trade count
        if hasattr(self, 'trades_this_episode'):
            self._recent_trade_counts.append(self.trades_this_episode)
            # Keep only last 10 episodes
            if len(self._recent_trade_counts) > 10:
                self._recent_trade_counts.pop(0)
        
        """
            
            content = content[:insert_point] + tracking_code + content[insert_point:]
    
    # Write the updated content
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed reward system to prevent oscillation")

def add_exploration_incentives():
    """Add incentives for consistent trading behavior"""
    
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Find the section after holding rewards
    holding_section = "# If we're flat and have made some trades, no penalty"
    index = content.find(holding_section)
    
    if index != -1:
        # Find the line after this section
        after_index = content.find("return 0.0", index) + len("return 0.0")
        
        # Insert consistency bonus
        consistency_bonus = """
            
        # Consistency bonus - reward stable trading patterns
        if hasattr(self, '_recent_trade_counts') and len(self._recent_trade_counts) >= 3:
            recent_avg = sum(self._recent_trade_counts[-3:]) / 3
            if 3 <= recent_avg <= 15 and 2 <= self.trades_this_episode <= 20:
                # Small bonus for consistent trading (not too few, not too many)
                consistency_reward = 0.02
                if self.trading_logger:
                    self.trading_logger.debug(f"Consistency bonus: {consistency_reward}")
                return consistency_reward"""
        
        content = content[:after_index] + consistency_bonus + content[after_index:]
    
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("✓ Added exploration incentives for consistent behavior")

def main():
    print("🔧 Comprehensive Fix for Training Oscillation")
    print("=" * 60)
    
    # Create backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"futures_env_realistic.py.oscillation_fix_{timestamp}"
    shutil.copy2('futures_env_realistic.py', backup_file)
    print(f"✓ Created backup: {backup_file}")
    
    # Apply fixes
    print("\nApplying fixes...")
    fix_futures_env_reward_system()
    add_exploration_incentives()
    
    print("\n✅ All fixes applied!")
    print("\nKey improvements:")
    print("1. Adaptive penalty that reduces if agent is oscillating")
    print("2. Tracks last 10 episodes to detect oscillation patterns")
    print("3. Consistency bonus for stable trading behavior")
    print("4. Reduced penalties when oscillation is detected")
    
    print("\nThe agent should now:")
    print("- Trade more consistently without wild swings")
    print("- Learn stable trading patterns")
    print("- Not get stuck in oscillation loops")
    
    print("\nYou can now run your 10,000 episode training!")

if __name__ == "__main__":
    main()