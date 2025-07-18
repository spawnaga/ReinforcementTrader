#!/usr/bin/env python3
"""
Balanced profitability fix that encourages selective but active trading
Goal: Achieve 40-60% profitability without excessive caution
"""

import re
from datetime import datetime

def apply_balanced_profitability_fix():
    print("🎯 Balanced Profitability Fix - Encouraging Smart Trading")
    print("=" * 60)
    
    # Read the current file
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Backup current version
    backup_name = f"futures_env_realistic.py.balanced_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_name, 'w') as f:
        f.write(content)
    print(f"✓ Created backup: {backup_name}")
    
    # Find the entire get_reward method and replace it
    reward_method_pattern = r'def get_reward\(self, state: TimeSeriesState\) -> float:.*?(?=\n    def|\n\nclass|\Z)'
    
    new_reward_method = '''def get_reward(self, state: TimeSeriesState) -> float:
        """Balanced reward system that encourages selective profitable trading"""
        
        # Track reward components for debugging
        reward_components = {}
        
        # 1. CLOSED POSITION REWARDS - Main learning signal
        if self.current_position == 0 and self.last_position != 0:
            # Calculate actual P&L
            if self.last_position == 1:  # Closed long
                gross_pnl = ((self._last_closed_exit_price - self._last_closed_entry_price) / self.tick_size) * self.value_per_tick
            else:  # Closed short
                gross_pnl = ((self._last_closed_entry_price - self._last_closed_exit_price) / self.tick_size) * self.value_per_tick
            
            # Subtract costs
            net_pnl = gross_pnl - (2 * self.execution_cost_per_order)
            base_reward = net_pnl
            reward_components['net_pnl'] = net_pnl
            
            # Quality multipliers for exceptional trades
            if net_pnl > 0:
                profit_pct = (net_pnl / (abs(self._last_closed_entry_price) * self.value_per_tick / self.tick_size)) * 100
                
                if profit_pct > 2.0:  # Exceptional trade (>2%)
                    multiplier = 1.5  # 50% bonus
                    base_reward *= multiplier
                    reward_components['exceptional_multiplier'] = (multiplier - 1) * net_pnl
                elif profit_pct > 1.0:  # Great trade (>1%)
                    multiplier = 1.3  # 30% bonus
                    base_reward *= multiplier
                    reward_components['great_multiplier'] = (multiplier - 1) * net_pnl
                elif profit_pct > 0.5:  # Good trade (>0.5%)
                    multiplier = 1.15  # 15% bonus
                    base_reward *= multiplier
                    reward_components['good_multiplier'] = (multiplier - 1) * net_pnl
                
                # Win streak bonus
                if hasattr(self, '_consecutive_wins'):
                    self._consecutive_wins += 1
                    if self._consecutive_wins >= 3:
                        streak_bonus = 10.0 * min(self._consecutive_wins - 2, 5)  # $10-50 bonus
                        base_reward += streak_bonus
                        reward_components['streak_bonus'] = streak_bonus
                else:
                    self._consecutive_wins = 1
            else:
                # Reset win streak
                self._consecutive_wins = 0
                
                # Reduce penalty for small losses (learning to minimize losses is important)
                if -50 < net_pnl < 0:
                    reduction = abs(net_pnl) * 0.2  # 20% reduction
                    base_reward += reduction
                    reward_components['small_loss_reduction'] = reduction
            
            # Log detailed reward breakdown
            if self.trading_logger:
                self.trading_logger.log_reward_calculation(
                    "CLOSED_POSITION",
                    timestamp=state.ts,
                    details={
                        'episode': self.episode_number,
                        'components': reward_components,
                        'total_reward': base_reward
                    }
                )
            
            return base_reward
        
        # 2. HOLDING REWARDS - Encourage good holds, discourage bad ones
        if self.current_position != 0 and self.entry_price is not None:
            current_price = state.price
            
            if self.current_position == 1:  # Long
                unrealized_pnl = ((current_price - self.entry_price) / self.tick_size) * self.value_per_tick
            else:  # Short
                unrealized_pnl = ((self.entry_price - current_price) / self.tick_size) * self.value_per_tick
            
            # Dynamic holding rewards based on P&L trajectory
            if unrealized_pnl > 50:
                # Profitable position - small reward to encourage holding winners
                hold_reward = min(unrealized_pnl * 0.002, 2.0)  # 0.2% of profit, max $2
                reward_components['profitable_hold'] = hold_reward
            elif unrealized_pnl > 0:
                # Small profit - tiny reward
                hold_reward = 0.1
                reward_components['small_profit_hold'] = hold_reward
            elif unrealized_pnl > -25:
                # Small loss - no reward or penalty (let agent decide)
                hold_reward = 0.0
            elif unrealized_pnl > -100:
                # Moderate loss - small penalty to encourage exit
                hold_reward = -0.5
                reward_components['moderate_loss_hold'] = hold_reward
            else:
                # Large loss - increasing penalty to teach stop losses
                hold_reward = min(-1.0 - (abs(unrealized_pnl) - 100) * 0.01, -5.0)
                reward_components['large_loss_hold'] = hold_reward
            
            # Time penalty for extremely long holds
            if self.holding_time > 100:
                time_penalty = -0.05 * min((self.holding_time - 100) / 100, 1.0)  # Max -0.05
                hold_reward += time_penalty
                reward_components['excessive_hold_penalty'] = time_penalty
            
            return hold_reward
        
        # 3. OPPORTUNITY COST SIGNAL (not penalty, but signal)
        # Encourage trading when volatility suggests opportunities
        if self.current_position == 0:
            # Calculate recent price volatility
            if hasattr(self, 'states') and len(self.states) > 20:
                recent_prices = [s.price for s in self.states[-20:]]
                price_range = max(recent_prices) - min(recent_prices)
                volatility = price_range / self.tick_size
                
                # High volatility + not trading = small opportunity cost
                if volatility > 10 and self.trades_this_episode < 5:
                    # Not a penalty, but a small negative to nudge towards action
                    opportunity_cost = -0.02  # Very small
                    
                    # But only in early/mid episodes when learning
                    if self.episode_number < 200:
                        reward_components['opportunity_signal'] = opportunity_cost
                        return opportunity_cost
            
            # Activity bonus for reasonable trading frequency
            if 2 <= self.trades_this_episode <= 10:
                activity_bonus = 0.01  # Tiny positive for being active
                reward_components['activity_bonus'] = activity_bonus
                return activity_bonus
        
        # Default: truly neutral (no reward, no penalty)
        return 0.0'''
    
    # Replace the entire get_reward method
    content = re.sub(reward_method_pattern, new_reward_method, content, flags=re.DOTALL)
    
    # Make sure we have the tracking variables
    reset_addition = '''
        # Track for balanced reward system
        self._consecutive_wins = 0
        if not hasattr(self, '_episode_profits'):
            self._episode_profits = []'''
    
    # Find reset method and add tracking
    reset_pattern = r'(def reset\(self.*?\n)([ ]+)(""".*?""")'
    reset_replacement = r'\1\2\3' + reset_addition
    content = re.sub(reset_pattern, reset_replacement, content, flags=re.DOTALL)
    
    # Write the fixed file
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("\n✅ Balanced profitability fix applied!")
    print("\nKey features:")
    print("1. ✓ Quality multipliers: 1.5x for >2% profit, 1.3x for >1%, 1.15x for >0.5%")
    print("2. ✓ Win streak bonuses: $10-50 for consecutive wins")
    print("3. ✓ Smart holding rewards: Positive for profits, negative for big losses")
    print("4. ✓ Opportunity signals: Tiny nudges during high volatility")
    print("5. ✓ Activity bonus: Small reward for reasonable trading frequency")
    print("\nExpected results:")
    print("- More active trading (not excessive)")
    print("- 40-60% profitability target")
    print("- Better win rates on trades taken")
    print("- No forced trading, just smart encouragement")

if __name__ == "__main__":
    apply_balanced_profitability_fix()