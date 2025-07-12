# Local Development Setup

## Quick Start

The system now uses PostgreSQL and loads configuration from a `.env` file.

### Option 1: Using .env file (Recommended)

1. The `.env` file has been created with your database credentials:
   ```
   DATABASE_URL=postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader
   ```

2. Simply run:
   ```bash
   python run_local.py
   ```

   The application will automatically load the DATABASE_URL from the .env file.

### Option 2: Export Environment Variable

If you prefer to set the environment variable manually:

```bash
# Set for current session only
export DATABASE_URL="postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader"
python run_local.py
```

### Option 3: Add to Shell Configuration

To make it permanent, add to your shell configuration file:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export DATABASE_URL="postgresql://trader_user:your_secure_password@localhost:5432/reinforcement_trader"' >> ~/.bashrc

# Reload
source ~/.bashrc
```

## Why the Change?

- **SQLite limitations**: SQLite only allows one writer at a time, causing "database is locked" errors during multi-GPU training
- **PostgreSQL benefits**: 
  - True concurrent access for multiple GPUs
  - Better performance for complex queries
  - Production-ready database
  - No more database locking issues

## Troubleshooting

### "DATABASE_URL not set" Error
- Make sure the `.env` file exists in your project root
- Check that python-dotenv is installed: `pip install python-dotenv`
- Verify the .env file contains: `DATABASE_URL=postgresql://...`

### Connection Refused
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify the connection details in your DATABASE_URL

### Authentication Failed
- Double-check your password in the DATABASE_URL
- Make sure the user has proper permissions

## Security Note

The `.env` file is already added to `.gitignore` to prevent accidentally committing your credentials.