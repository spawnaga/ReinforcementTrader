#!/usr/bin/env python3
"""
Fix all instances where trading_logger is used without null checks
"""

import re

def fix_trading_logger_null_checks():
    """Add null checks to all trading_logger calls"""
    
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Pattern to find trading_logger calls that are not already protected
    # We need to find lines that have self.trading_logger. but NOT preceded by "if self.trading_logger"
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains an unprotected trading_logger call
        if 'self.trading_logger.' in line and 'if self.trading_logger' not in line:
            # Check if the previous line already has the null check
            prev_line = lines[i-1] if i > 0 else ""
            
            # If not already protected, add protection
            if 'if self.trading_logger:' not in prev_line:
                # Get the indentation of the current line
                indent = len(line) - len(line.lstrip())
                
                # Check if this is part of a multi-line statement
                if line.strip().endswith('('):
                    # This is the start of a multi-line call
                    # Find the end of the multi-line statement
                    j = i + 1
                    while j < len(lines) and ')' not in lines[j]:
                        j += 1
                    
                    # Add the if statement
                    fixed_lines.append(' ' * indent + 'if self.trading_logger:')
                    
                    # Add all lines of the multi-line statement with extra indentation
                    for k in range(i, j + 1):
                        fixed_lines.append('    ' + lines[k])
                    
                    # Skip to after the multi-line statement
                    i = j + 1
                    continue
                else:
                    # Single line statement
                    fixed_lines.append(' ' * indent + 'if self.trading_logger:')
                    fixed_lines.append('    ' + line)
                    i += 1
                    continue
        
        # Normal case - just add the line
        fixed_lines.append(line)
        i += 1
    
    # Write the fixed content
    with open('futures_env_realistic.py', 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("✓ Fixed all trading_logger null checks")
    
    # Count how many fixes were made
    original_count = content.count('self.trading_logger.')
    protected_count = content.count('if self.trading_logger:')
    
    print(f"  - Total trading_logger calls: {original_count}")
    print(f"  - Previously protected: {protected_count}")
    print(f"  - Newly protected: {original_count - protected_count}")

def verify_fix():
    """Verify that all trading_logger calls are now protected"""
    with open('futures_env_realistic.py', 'r') as f:
        lines = f.readlines()
    
    unprotected = []
    for i, line in enumerate(lines):
        if 'self.trading_logger.' in line and 'if self.trading_logger' not in line:
            # Check if previous line has the protection
            if i > 0 and 'if self.trading_logger:' not in lines[i-1]:
                unprotected.append((i+1, line.strip()))
    
    if unprotected:
        print("\n⚠️ Warning: Some trading_logger calls may still be unprotected:")
        for line_num, line in unprotected[:5]:  # Show first 5
            print(f"  Line {line_num}: {line}")
    else:
        print("\n✅ All trading_logger calls are now protected with null checks!")

def main():
    print("🔧 Fixing Trading Logger Null Checks")
    print("=" * 60)
    
    # Create backup
    import shutil
    shutil.copy2('futures_env_realistic.py', 'futures_env_realistic.py.bak')
    print("✓ Created backup: futures_env_realistic.py.bak")
    
    # Apply fixes
    fix_trading_logger_null_checks()
    
    # Verify
    verify_fix()
    
    print("\nYou can now run your training again without AttributeError!")

if __name__ == "__main__":
    main()