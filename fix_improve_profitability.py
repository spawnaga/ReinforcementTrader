#!/usr/bin/env python3
"""
Fix to improve agent profitability from 27-28% to 50%+
Main changes:
1. Remove penalties for not trading (let agent be selective)
2. Add rewards for holding profitable positions
3. Remove reward scaling that caps big wins
4. Extend curriculum learning to 500 episodes
5. Add win rate bonus to encourage higher quality trades
"""

import re
from datetime import datetime

def apply_profitability_fixes():
    print("🎯 Improving Agent Profitability")
    print("=" * 60)
    
    # Read the current file
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Backup current version
    backup_name = f"futures_env_realistic.py.before_profitability_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_name, 'w') as f:
        f.write(content)
    print(f"✓ Created backup: {backup_name}")
    
    # Fix 1: Change penalty system - remove penalties for not trading
    # Find the penalty section
    penalty_pattern = r'# Check for oscillation and apply adaptive penalty.*?return -penalty'
    
    new_penalty_code = '''# Check for oscillation but DON'T penalize not trading
        # Let the agent be selective about when to trade
        
        # Track episode history for pattern detection
        if hasattr(self, 'episode_history'):
            self.episode_history.append(self.trades_this_episode)
            if len(self.episode_history) > 10:
                self.episode_history.pop(0)
                
            # Detect oscillation pattern
            if len(self.episode_history) == 10:
                pattern1 = all(x > 0 if i % 2 == 0 else x == 0 for i, x in enumerate(self.episode_history[-6:]))
                pattern2 = all(x == 0 if i % 2 == 0 else x > 0 for i, x in enumerate(self.episode_history[-6:]))
                
                if pattern1 or pattern2:
                    if self.trading_logger:
                        self.trading_logger.warning(
                            f"Oscillation detected but NOT penalizing - let agent choose when to trade"
                        )
        
        # NO PENALTY for not trading - this is crucial!
        # The agent should only trade when it sees good opportunities
        return 0.0'''
    
    content = re.sub(penalty_pattern, new_penalty_code, content, flags=re.DOTALL)
    
    # Fix 2: Add reward for holding profitable positions
    hold_reward_pattern = r'# Hold position reward.*?return hold_reward'
    
    new_hold_reward = '''# Hold position reward - encourage holding winners
        if self.current_position != 0:
            # Calculate unrealized P&L
            current_price = state.price
            if self.current_position == 1:  # Long position
                unrealized_pnl = ((current_price - self.entry_price) / self.tick_size) * self.value_per_tick
            else:  # Short position
                unrealized_pnl = ((self.entry_price - current_price) / self.tick_size) * self.value_per_tick
            
            # Progressive reward for profitable positions
            if unrealized_pnl > 0:
                # Reward holding winners: small continuous reward
                # This teaches the agent to let profits run
                hold_reward = min(unrealized_pnl * 0.01, 5.0)  # 1% of unrealized profit, max $5/step
                
                # Extra reward for holding through multiple profitable periods
                if self.holding_time > 5 and unrealized_pnl > 50:
                    hold_reward += 2.0  # Bonus for patience
                
                if self.trading_logger and self.episode_number % 100 == 0:
                    self.trading_logger.debug(
                        f"Rewarding profitable hold: unrealized_pnl=${unrealized_pnl:.2f}, reward=${hold_reward:.2f}"
                    )
                return hold_reward
            elif unrealized_pnl < -100:
                # Small penalty for holding big losers too long
                # This teaches the agent to cut losses
                return -2.0
            else:
                # Small negative reward for holding losing positions
                # But not too harsh - sometimes drawdown is temporary
                return max(unrealized_pnl * 0.005, -1.0)  # 0.5% of loss, max -$1/step
        
        return 0.0'''
    
    content = re.sub(hold_reward_pattern, new_hold_reward, content, flags=re.DOTALL)
    
    # Fix 3: Remove reward scaling that limits learning from big wins
    scaling_pattern = r'# CRITICAL FIX: Scale down large rewards.*?base_reward = scaled_reward'
    
    new_scaling = '''# Allow full rewards - don't cap successful trades!
            # The agent needs to experience the full benefit of great trades
            # to learn what works. Only cap extreme outliers.
            if abs(base_reward) > 1000:
                # Only scale truly extreme rewards to prevent instability
                sign = 1 if base_reward > 0 else -1
                scaled_reward = sign * (1000 + 10 * np.log(abs(base_reward) / 1000))
                
                if self.trading_logger:
                    self.trading_logger.warning(
                        f"EXTREME REWARD SCALING: ${base_reward:.2f} scaled to ${scaled_reward:.2f}"
                    )
                
                base_reward = scaled_reward'''
    
    content = re.sub(scaling_pattern, new_scaling, content, flags=re.DOTALL)
    
    # Fix 4: Extend curriculum learning to 500 episodes
    content = content.replace('if self.episode_number < 50:', 'if self.episode_number < 200:')
    content = content.replace('elif self.episode_number < 150:', 'elif self.episode_number < 500:')
    
    # Fix 5: Better reward shaping for quality trades
    easy_stage_pattern = r'if base_reward > 0:.*?fixed_bonus = 5\.0.*?return total_reward'
    
    new_easy_stage = '''if base_reward > 0:
                    # Reward based on trade quality, not just profit
                    profit_ratio = base_reward / (abs(self._last_closed_entry_price) * 0.01)  # Profit as % of entry
                    
                    if profit_ratio > 0.5:  # > 0.5% profit
                        quality_bonus = 10.0  # Great trade
                    elif profit_ratio > 0.2:  # > 0.2% profit
                        quality_bonus = 5.0   # Good trade
                    else:
                        quality_bonus = 2.0   # Small profit
                    
                    total_reward = base_reward + quality_bonus
                    
                    if self.trading_logger:
                        self.trading_logger.log_reward_calculation(
                            "QUALITY TRADE BONUS",
                            timestamp=state.ts,
                            details={
                                'episode': self.episode_number,
                                'base_reward': base_reward,
                                'profit_ratio': profit_ratio,
                                'quality_bonus': quality_bonus,
                                'total_reward': total_reward
                            }
                        )
                    return total_reward'''
    
    content = re.sub(easy_stage_pattern, new_easy_stage, content, flags=re.DOTALL)
    
    # Write the fixed file
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("\n✅ All profitability improvements applied!")
    print("\nKey improvements:")
    print("1. ✓ Removed penalties for not trading - agent can be selective")
    print("2. ✓ Added rewards for holding profitable positions")
    print("3. ✓ Removed reward scaling that was capping big wins")
    print("4. ✓ Extended curriculum learning to 500 episodes")
    print("5. ✓ Added quality-based bonuses for good trades")
    print("\nExpected outcome:")
    print("- Profitability should improve from 27-28% to 50%+")
    print("- Agent will be more selective about trades")
    print("- Agent will hold winners longer and cut losers faster")
    print("- Less oscillation between trading/not trading")
    print("\nRun training again to see improved performance!")

if __name__ == "__main__":
    apply_profitability_fixes()