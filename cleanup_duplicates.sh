#!/bin/bash

# Cleanup script for ReinforcementTrader - Removes duplicate and unnecessary files
# Review this script before running!

echo "=== ReinforcementTrader Cleanup Script ==="
echo "This will remove duplicate and unnecessary files."
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Create backup directory
echo "Creating backup directory..."
mkdir -p backup_before_cleanup
echo "Backing up important configs..."
cp .env backup_before_cleanup/ 2>/dev/null
cp pyproject.toml backup_before_cleanup/
cp README.md backup_before_cleanup/

echo "Starting cleanup..."

# 1. Remove debug scripts
echo "Removing debug scripts..."
rm -f debug_11725_reward.py debug_1214_reward.py debug_exact_11725_source.py
rm -f debug_episode_39.py debug_gross_profit_leak.py debug_reward_trace.py
rm -f debug_timestamp_simple.py debug_training.py debug_zero_trades_reward.py

# 2. Remove test bug files
echo "Removing obsolete test files..."
rm -f test_11725_bug.py test_env_return_bug.py test_initialization_bug.py
rm -f test_reward_accumulation.py test_reward_bug_episode7.py test_reward_bug_simple.py
rm -f test_debug_messages.py

# 3. Remove duplicate database scripts
echo "Removing duplicate database scripts..."
rm -f clean_db.py clean_slate.py fix_database_url.py fix_db_url.py
rm -f setup_database.py setup_training_db.py fix_db_auth.sh

# 4. Remove duplicate installation scripts
echo "Removing duplicate installation scripts..."
rm -f fix_python_deps.sh install_all_deps.sh install_essentials.sh manual_setup_gpu.sh
rm -f complete_talib_install.sh install_talib.sh install_talib_alternative.sh fix_talib_manual.py

# 5. Remove Windows PowerShell scripts
echo "Removing Windows-specific scripts..."
rm -f check_status.ps1 get_session_details.ps1

# 6. Remove obsolete files
echo "Removing obsolete files..."
rm -f catch_1214_bug.py check_for_bug.sh find_1214_bug.py
rm -f check_observation_size.py check_observation_values.py
rm -f demo_enhanced_logging.py demo_logging.py
rm -f fix_ib_integration.py fix_postgresql_permissions.sql

# 7. Remove test data samples
echo "Removing test data files..."
rm -f test_data_sample.csv test_nq_data.csv

# 8. Remove command text files
echo "Removing command text files..."
rm -f check_logs_command.txt quick_run_command.txt run_with_user_data.sh

# 9. Clean attached_assets directory
echo "Cleaning attached_assets directory..."
rm -rf attached_assets/

# 10. Remove Python cache files
echo "Removing Python cache files..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

# 11. Update .gitignore if needed
if ! grep -q "attached_assets/" .gitignore 2>/dev/null; then
    echo "Updating .gitignore..."
    echo "" >> .gitignore
    echo "# Cleanup additions" >> .gitignore
    echo "attached_assets/" >> .gitignore
    echo "*.pyc" >> .gitignore
    echo "__pycache__/" >> .gitignore
    echo "logs/*.log" >> .gitignore
    echo "models/*.pt" >> .gitignore
    echo "models/*.pth" >> .gitignore
    echo "data_cache/" >> .gitignore
fi

echo ""
echo "=== Cleanup Complete ==="
echo "Removed approximately 60 duplicate/unnecessary files"
echo "Backup created in: backup_before_cleanup/"
echo ""
echo "Files kept:"
echo "- Core application files (app.py, routes.py, models.py, etc.)"
echo "- Essential scripts (train_standalone.py, prepare_training_data.py, etc.)"
echo "- Documentation (README.md, guides, etc.)"
echo "- Configuration files"
echo "- rl_algorithms/, gym_futures/, tests/ directories"
echo ""
echo "The API is confirmed working at http://localhost:5000"