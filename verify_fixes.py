#!/usr/bin/env python3
"""
Verify that all database type conversion fixes are in place
"""
import sys
import ast

def check_file_for_fixes(filepath):
    """Check if the file has the proper type conversions"""
    print(f"\nChecking {filepath}...")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    issues = []
    
    # Check for Decimal conversion in get_learning_assessment
    if 'get_learning_assessment' in content:
        if 'float(comparison[' in content:
            print("✓ get_learning_assessment has Decimal->float conversion")
        else:
            issues.append("✗ get_learning_assessment missing Decimal conversion")
    
    # Check for numpy type conversions
    if 'int(self.current_episode)' in content:
        print("✓ Episode numbers are converted to int")
    else:
        issues.append("✗ Episode numbers not converted to int")
        
    if 'float(total_reward)' in content:
        print("✓ Rewards are converted to float")
    else:
        issues.append("✗ Rewards not converted to float")
    
    # Check for proper cursor usage in _update_learning_progress
    if '_update_learning_progress' in content:
        if 'e[0]' in content and 'e[1]' in content:
            print("✓ _update_learning_progress uses tuple indexing correctly")
        else:
            issues.append("✗ _update_learning_progress has incorrect indexing")
    
    return issues

def main():
    print("Verifying database type conversion fixes...")
    print("=" * 60)
    
    issues = check_file_for_fixes('training_tracker.py')
    
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nPlease ensure you've saved all changes and restarted Python.")
        sys.exit(1)
    else:
        print("\n✅ All fixes are in place!")
        print("\nIf you're still getting errors, try:")
        print("1. Exit Python completely")
        print("2. Run: python -m py_compile training_tracker.py")
        print("3. Start your training again")
        print("\nOr simply run in a new terminal:")
        print("  python train_standalone.py --num-gpus 4 --episodes 10000")

if __name__ == "__main__":
    main()