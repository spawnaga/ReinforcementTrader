#!/bin/bash
# Script to check for the 11,735 reward bug in logs

echo "=== Checking for 11,735 Reward Bug ==="
echo

# Check performance log for episode rewards
echo "1. Checking episode rewards..."
grep -n "Episode" logs/latest/performance.log 2>/dev/null | grep -E "(11[0-9]{3}|Reward.*11)" | head -10

echo
echo "2. Checking algorithm log for high rewards..."
grep -n "Episode\|Reward" logs/latest/algorithm.log 2>/dev/null | grep -E "(11[0-9]{3}|Episode [0-9]+ .*Reward)" | head -20

echo
echo "3. Checking rewards.log for debug messages..."
grep -E "(GET_REWARD|STEP REWARD|NO TRADE|REWARD SCALING|HUGE HOLD|11[0-9]{3})" logs/latest/rewards.log 2>/dev/null | head -20

echo
echo "4. Checking for trades in episodes with high rewards..."
grep -B2 -A2 "11[0-9]{3}" logs/latest/*.log 2>/dev/null | head -30

echo
echo "5. Summary of all episodes..."
grep "Episode.*Reward" logs/latest/performance.log 2>/dev/null | awk '{print $2, $4, $6, $8, $10}' | column -t

echo
echo "=== Debug Messages ==="
grep -h -E "(GET_REWARD CALLED|STEP REWARD TRACE|NO TRADE PENALTY|REWARD SCALING|HUGE HOLD REWARD)" logs/latest/*.log 2>/dev/null | head -20
if [ $? -ne 0 ]; then
    echo "No debug messages found. The bug might not have occurred in this run."
fi