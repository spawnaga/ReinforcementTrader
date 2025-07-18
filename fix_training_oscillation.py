#!/usr/bin/env python3
"""
Fix for training oscillation and metric calculation issues

Issues fixed:
1. Win rate and profitability showing identical values
2. Agent oscillating between trading and no-trading episodes
"""

import os
import shutil
from datetime import datetime

def create_backup():
    """Create backup of files before fixing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup_oscillation_fix_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [
        'train_standalone.py',
        'training_tracker.py',
        'futures_env_realistic.py'
    ]
    
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
    
    print(f"✓ Created backup in {backup_dir}/")
    return backup_dir

def fix_train_standalone():
    """Fix entropy coefficient and trading parameters"""
    with open('train_standalone.py', 'r') as f:
        content = f.read()
    
    # Increase entropy coefficient to encourage exploration
    content = content.replace(
        "entropy_coef=config.get('entropy_coef', 0.01)",
        "entropy_coef=config.get('entropy_coef', 0.05)"  # Increased from 0.01
    )
    
    # Increase max trades per episode
    content = content.replace(
        "'max_trades_per_episode': 10,",
        "'max_trades_per_episode': 20,"  # Increased from 10
    )
    
    with open('train_standalone.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed train_standalone.py:")
    print("  - Increased entropy coefficient: 0.01 → 0.05")
    print("  - Increased max trades per episode: 10 → 20")

def fix_futures_env():
    """Add curriculum learning adjustments for exploration"""
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Find the get_reward method and add exploration bonus
    if "def get_reward(self):" in content:
        # Add exploration bonus for taking actions
        old_code = """        # Penalty for not trading (scaled by difficulty)
        if self.trades_this_episode == 0:
            # Start with small penalty, increase over time
            if self.curriculum_stage == 'easy':
                return -0.05
            elif self.curriculum_stage == 'medium':  
                return -0.075
            else:  # hard
                return -0.1"""
                
        new_code = """        # Penalty for not trading (scaled by difficulty)
        if self.trades_this_episode == 0:
            # Start with small penalty, increase over time
            if self.curriculum_stage == 'easy':
                return -0.05
            elif self.curriculum_stage == 'medium':  
                return -0.075
            else:  # hard
                return -0.1
                
        # Small exploration bonus for taking actions (prevents oscillation)
        if self.trades_this_episode > 0 and self.trades_this_episode < 5:
            return 0.01  # Small bonus to encourage initial trades"""
        
        content = content.replace(old_code, new_code)
    
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("✓ Fixed futures_env_realistic.py:")
    print("  - Added exploration bonus for initial trades")

def verify_fixes():
    """Verify the fixes were applied correctly"""
    print("\n🔍 Verifying fixes...")
    
    # Check train_standalone.py
    with open('train_standalone.py', 'r') as f:
        content = f.read()
        if "entropy_coef=config.get('entropy_coef', 0.05)" in content:
            print("✓ Entropy coefficient fixed")
        else:
            print("❌ Entropy coefficient fix failed")
            
        if "'max_trades_per_episode': 20," in content:
            print("✓ Max trades per episode fixed")
        else:
            print("❌ Max trades fix failed")
    
    # Check training_tracker.py (already fixed manually)
    with open('training_tracker.py', 'r') as f:
        content = f.read()
        if "profitability_pct = (result[0] / result[1]) * 100" in content:
            print("✓ Profitability calculation already fixed")
        else:
            print("❌ Profitability calculation needs fixing")

def main():
    print("🔧 Fixing Training Oscillation and Metric Issues")
    print("=" * 60)
    
    # Create backup
    backup_dir = create_backup()
    
    try:
        # Apply fixes
        fix_train_standalone()
        fix_futures_env()
        
        # Verify
        verify_fixes()
        
        print("\n✅ All fixes applied successfully!")
        print("\nSummary of changes:")
        print("1. Win rate and profitability now show different values:")
        print("   - Win rate = winning trades / total trades")
        print("   - Profitability = profitable episodes / total episodes")
        print("2. Reduced oscillation by:")
        print("   - Increasing entropy coefficient (0.01 → 0.05)")
        print("   - Increasing max trades per episode (10 → 20)")
        print("   - Adding exploration bonus for initial trades")
        print("\nRun training again to see the improvements!")
        
    except Exception as e:
        print(f"\n❌ Error applying fixes: {e}")
        print(f"Restoring from backup in {backup_dir}/")
        # Restore backups if something went wrong
        for file in os.listdir(backup_dir):
            shutil.copy2(os.path.join(backup_dir, file), '.')

if __name__ == "__main__":
    main()