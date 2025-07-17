#!/bin/bash
# AI Trading System - File Cleanup Commands
# This script removes duplicate and unnecessary files

echo "🧹 AI Trading System - File Cleanup"
echo "This will remove 12 duplicate/unnecessary files"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Delete duplicate entry points
echo "Removing duplicate entry points..."
rm -f main.py

# Delete duplicate monitoring scripts
echo "Removing duplicate monitoring scripts..."
rm -f monitor.py
rm -f gpu_monitor.py
rm -f check_training_status.py

# Delete duplicate database scripts
echo "Removing duplicate database scripts..."
rm -f clean_db.py
rm -f cleanup_database.sql
rm -f fix_postgresql_permissions.sql

# Delete demo scripts (already in README)
echo "Removing demo scripts..."
rm -f demo_cli.py
rm -f demo_gpu_training.py

# Delete PowerShell scripts
echo "Removing PowerShell scripts..."
rm -f check_status.ps1
rm -f get_session_details.ps1

# Delete duplicate environment file
echo "Removing duplicate environment file..."
rm -f futures_env_realistic.py

echo "✅ Cleanup complete! Removed 12 files."
echo ""
echo "Next steps:"
echo "1. Merge extensions.py into app.py"
echo "2. Merge trading_config.py into config.py"
echo "3. Update any imports that reference deleted files"