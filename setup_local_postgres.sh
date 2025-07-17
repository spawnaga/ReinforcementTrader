#!/bin/bash
# Setup PostgreSQL for local training

echo "Setting up PostgreSQL for multi-GPU training..."

# Create database and user
echo "Enter PostgreSQL password for trader_user:"
read -s DB_PASSWORD

sudo -u postgres psql << EOF
CREATE USER trader_user WITH PASSWORD '$DB_PASSWORD';
CREATE DATABASE reinforcement_trader OWNER trader_user;
GRANT ALL PRIVILEGES ON DATABASE reinforcement_trader TO trader_user;
EOF

# Create .env file with database URL
echo "# Database configuration" > .env
echo "DATABASE_URL=postgresql://trader_user:$DB_PASSWORD@localhost:5432/reinforcement_trader" >> .env
echo "" >> .env
echo "# Optional: Add your own API keys here" >> .env
echo "# OPENAI_API_KEY=your_key_here" >> .env

echo "✓ PostgreSQL setup complete!"
echo "✓ .env file created with DATABASE_URL"

# Setup database tables
echo ""
echo "Setting up database tables..."
python setup_training_db.py

echo ""
echo "Database ready for training!"