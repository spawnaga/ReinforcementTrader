#!/usr/bin/env python3
"""Fix database URL encoding"""
from urllib.parse import quote_plus

# Original password with special characters
password = "P@Gr&Dt0y%WA2q!B"

# Properly URL encode the password
encoded_password = quote_plus(password)

print(f"Original password: {password}")
print(f"Encoded password: {encoded_password}")
print(f"\nProperly encoded DATABASE_URL:")
print(f"DATABASE_URL=postgresql://trader_user:{encoded_password}@localhost:5432/reinforcement_trader")