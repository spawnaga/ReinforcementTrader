#!/usr/bin/env python3
"""Fix the ContFut error in ib_integration.py"""

# Read the file
with open('ib_integration.py', 'r') as f:
    content = f.read()

# Replace ContFut with Future
content = content.replace('self.nq_contract = ContFut(', 'self.nq_contract = Future(')

# Write back
with open('ib_integration.py', 'w') as f:
    f.write(content)

print("Fixed ContFut -> Future in ib_integration.py")