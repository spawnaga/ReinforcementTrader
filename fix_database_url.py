#!/usr/bin/env python3
"""Fix DATABASE_URL with proper URL encoding"""

import urllib.parse
import os

# Read current .env file
with open('.env', 'r') as f:
    lines = f.readlines()

# Fix the DATABASE_URL
new_lines = []
for line in lines:
    if line.startswith('DATABASE_URL='):
        # Parse the current URL
        parts = line.strip().split('=', 1)[1]
        # Extract password from malformed URL
        # Current: postgresql://trader_user:P@Gr&Dt0y%WA2q!B@localhost:5432/reinforcement_trader
        
        user = 'trader_user'
        password = 'P@Gr&Dt0y%WA2q!B'  # The generated password
        host = 'localhost'
        port = '5432'
        database = 'reinforcement_trader'
        
        # URL encode the password
        encoded_password = urllib.parse.quote(password, safe='')
        
        # Build proper URL
        new_url = f"postgresql://{user}:{encoded_password}@{host}:{port}/{database}"
        new_lines.append(f'DATABASE_URL={new_url}\n')
        print(f"Fixed DATABASE_URL with properly encoded password")
    else:
        new_lines.append(line)

# Write back
with open('.env', 'w') as f:
    f.writelines(new_lines)

print("Updated .env file with properly encoded DATABASE_URL")