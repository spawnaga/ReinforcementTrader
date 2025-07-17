#!/usr/bin/env python3
"""Fix the trading engine syntax error"""

# Read the file
with open('trading_engine.py', 'r') as f:
    lines = f.readlines()

# Find and fix the duplicate line
fixed_lines = []
skip_next = False
for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
        
    # Check if this is the duplicate line
    if i < len(lines) - 1:
        current_stripped = line.strip()
        next_stripped = lines[i+1].strip()
        
        # If both lines contain total_episodes
        if 'total_episodes=config.get' in current_stripped and 'total_episodes=config.get' in next_stripped:
            fixed_lines.append(line)  # Keep only the first one
            skip_next = True  # Skip the next duplicate
            print(f"Found and fixed duplicate at line {i+1}")
            continue
    
    fixed_lines.append(line)

# Write back
with open('trading_engine.py', 'w') as f:
    f.writelines(fixed_lines)

print("Fixed trading_engine.py successfully!")