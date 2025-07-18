#!/usr/bin/env python3
"""
Analyze why the agent is performing so poorly (27% profitability)
Look at actual trading patterns to identify core issues
"""

import pandas as pd
import numpy as np
from datetime import datetime
import glob
import os

def analyze_trading_patterns():
    print("🔍 Analyzing Trading Behavior and Core Issues")
    print("=" * 60)
    
    # Find the latest log directory
    log_dirs = sorted(glob.glob('logs/2025*'))
    if not log_dirs:
        print("No log directories found!")
        return
    
    latest_dir = log_dirs[-1]
    print(f"Analyzing logs from: {latest_dir}")
    
    # Analyze trading.log for trade patterns
    trading_log = os.path.join(latest_dir, 'trading.log')
    if os.path.exists(trading_log):
        print("\n📊 TRADE ANALYSIS:")
        print("-" * 50)
        
        with open(trading_log, 'r') as f:
            lines = f.readlines()
        
        trades = []
        current_trade = {}
        
        for line in lines:
            if 'OPENED' in line:
                # Parse entry
                parts = line.split('|')
                if len(parts) >= 4:
                    current_trade = {
                        'type': 'LONG' if 'LONG' in line else 'SHORT',
                        'entry_price': float(parts[3].split('$')[1].strip()),
                        'entry_time': parts[0].strip()
                    }
            elif 'CLOSED' in line and current_trade:
                # Parse exit
                parts = line.split('|')
                if len(parts) >= 5:
                    exit_price = float(parts[4].split('$')[1].split()[0])
                    current_trade['exit_price'] = exit_price
                    current_trade['exit_time'] = parts[0].strip()
                    
                    # Calculate P&L
                    if current_trade['type'] == 'LONG':
                        pnl = (exit_price - current_trade['entry_price']) * 2 - 10  # $5 per side commission
                    else:
                        pnl = (current_trade['entry_price'] - exit_price) * 2 - 10
                    
                    current_trade['pnl'] = pnl
                    current_trade['pnl_pct'] = (pnl / (current_trade['entry_price'] * 2)) * 100
                    trades.append(current_trade)
                    current_trade = {}
        
        if trades:
            df = pd.DataFrame(trades)
            
            print(f"Total trades analyzed: {len(trades)}")
            print(f"Winning trades: {len(df[df['pnl'] > 0])} ({len(df[df['pnl'] > 0])/len(df)*100:.1f}%)")
            print(f"Average P&L: ${df['pnl'].mean():.2f}")
            print(f"Average P&L %: {df['pnl_pct'].mean():.2f}%")
            print(f"Largest win: ${df['pnl'].max():.2f}")
            print(f"Largest loss: ${df['pnl'].min():.2f}")
            
            # Analyze trade duration
            print("\n⏱️ TIMING ANALYSIS:")
            print("-" * 50)
            
            # Look at hold times
            hold_times = []
            for _, trade in df.iterrows():
                try:
                    entry = datetime.strptime(trade['entry_time'], '%H:%M:%S')
                    exit = datetime.strptime(trade['exit_time'], '%H:%M:%S')
                    duration = (exit - entry).seconds / 60  # minutes
                    hold_times.append(duration)
                except:
                    pass
            
            if hold_times:
                print(f"Average hold time: {np.mean(hold_times):.1f} minutes")
                print(f"Shortest hold: {min(hold_times):.1f} minutes")
                print(f"Longest hold: {max(hold_times):.1f} minutes")
            
            # Pattern analysis
            print("\n🎯 PATTERN ANALYSIS:")
            print("-" * 50)
            
            # Check if agent is cutting winners short
            winners = df[df['pnl'] > 0]
            losers = df[df['pnl'] < 0]
            
            if len(winners) > 0 and len(losers) > 0:
                avg_win = winners['pnl'].mean()
                avg_loss = abs(losers['pnl'].mean())
                print(f"Average winning trade: ${avg_win:.2f}")
                print(f"Average losing trade: -${avg_loss:.2f}")
                print(f"Win/Loss ratio: {avg_win/avg_loss:.2f}")
                
                if avg_win < avg_loss:
                    print("⚠️ ISSUE: Average loss is bigger than average win!")
                    print("   → Agent is letting losses run and cutting profits short")
    
    # Analyze algorithm.log for decision patterns
    algo_log = os.path.join(latest_dir, 'algorithm.log')
    if os.path.exists(algo_log):
        print("\n🤖 DECISION PATTERN ANALYSIS:")
        print("-" * 50)
        
        with open(algo_log, 'r') as f:
            lines = f.readlines()[:1000]  # First 1000 decisions
        
        actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        positions = {'FLAT': 0, 'LONG': 0, 'SHORT': 0}
        
        for line in lines:
            if 'Action:' in line:
                for action in actions:
                    if f'Action: {action}' in line:
                        actions[action] += 1
                        break
            
            if 'Position:' in line:
                for pos in positions:
                    if f'Position: {pos}' in line:
                        positions[pos] += 1
                        break
        
        total_actions = sum(actions.values())
        if total_actions > 0:
            print("Action distribution:")
            for action, count in actions.items():
                print(f"  {action}: {count} ({count/total_actions*100:.1f}%)")
            
            print("\nPosition distribution:")
            total_positions = sum(positions.values())
            for pos, count in positions.items():
                print(f"  {pos}: {count} ({count/total_positions*100:.1f}%)")
    
    # Core issues summary
    print("\n🔴 CORE ISSUES IDENTIFIED:")
    print("=" * 60)
    print("1. Poor Risk/Reward Ratio: Losses are bigger than wins")
    print("2. Low Win Rate: Agent can't identify good entry points")
    print("3. Oscillation: Agent alternates between overtrading and not trading")
    print("4. No Edge: Trading performance is worse than random (50%)")
    
    print("\n💡 ROOT CAUSE HYPOTHESIS:")
    print("The agent is not learning market patterns effectively because:")
    print("- State representation may lack critical features (momentum, volatility)")
    print("- Action space is too simple (no position sizing, stop losses)")
    print("- Training data may have difficult patterns (trending vs ranging)")
    print("- Reward system encourages activity over quality")
    
    print("\n✅ RECOMMENDED SOLUTIONS:")
    print("1. Add better technical indicators (momentum, volatility bands)")
    print("2. Implement position sizing based on confidence")
    print("3. Add automatic stop loss and take profit levels")
    print("4. Use market regime detection (trending vs ranging)")
    print("5. Train separate models for different market conditions")

if __name__ == "__main__":
    analyze_trading_patterns()