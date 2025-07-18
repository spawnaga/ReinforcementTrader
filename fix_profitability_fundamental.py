#!/usr/bin/env python3
"""
Fundamental fix for profitability - addressing root causes, not just rewards
Key insight: Agent loses more on losing trades than it wins on winning trades
Solution: Better risk management and market understanding
"""

import re
from datetime import datetime

def apply_fundamental_profitability_fix():
    print("🎯 Fundamental Profitability Fix - Risk Management & Market Understanding")
    print("=" * 60)
    
    # Read the current file
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Backup current version
    backup_name = f"futures_env_realistic.py.fundamental_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_name, 'w') as f:
        f.write(content)
    print(f"✓ Created backup: {backup_name}")
    
    # 1. Add stop loss and take profit mechanism
    # Find the __init__ method and add risk management parameters
    init_pattern = r'(self\.holding_time = 0)'
    init_addition = '''
        
        # Risk management parameters
        self.stop_loss_ticks = 10  # Stop loss at 10 ticks ($25)
        self.take_profit_ticks = 20  # Take profit at 20 ticks ($50)
        self.trailing_stop_active = False
        self.trailing_stop_high = None
        self.max_drawdown_ticks = 5  # Trail stop 5 ticks from high'''
    
    content = re.sub(init_pattern, r'\1' + init_addition, content)
    
    # 2. Enhance step method to implement stop loss and take profit
    step_enhancement = '''
        # Risk management: Check stop loss and take profit
        if self.current_position != 0 and self.entry_price is not None:
            current_price = state.price
            
            if self.current_position == 1:  # Long position
                # Calculate unrealized P&L in ticks
                unrealized_ticks = (current_price - self.entry_price) / self.tick_size
                
                # Take profit
                if unrealized_ticks >= self.take_profit_ticks:
                    self.sell(state)
                    if self.trading_logger:
                        self.trading_logger.info("Take profit triggered for long position")
                
                # Stop loss
                elif unrealized_ticks <= -self.stop_loss_ticks:
                    self.sell(state)
                    if self.trading_logger:
                        self.trading_logger.info("Stop loss triggered for long position")
                
                # Trailing stop
                elif unrealized_ticks > 10:  # Activate after 10 ticks profit
                    if not self.trailing_stop_active:
                        self.trailing_stop_active = True
                        self.trailing_stop_high = current_price
                    else:
                        # Update trailing stop high
                        if current_price > self.trailing_stop_high:
                            self.trailing_stop_high = current_price
                        # Check if we've pulled back too much
                        elif (self.trailing_stop_high - current_price) / self.tick_size > self.max_drawdown_ticks:
                            self.sell(state)
                            if self.trading_logger:
                                self.trading_logger.info("Trailing stop triggered for long position")
            
            else:  # Short position
                # Calculate unrealized P&L in ticks
                unrealized_ticks = (self.entry_price - current_price) / self.tick_size
                
                # Take profit
                if unrealized_ticks >= self.take_profit_ticks:
                    self.buy(state)
                    if self.trading_logger:
                        self.trading_logger.info("Take profit triggered for short position")
                
                # Stop loss
                elif unrealized_ticks <= -self.stop_loss_ticks:
                    self.buy(state)
                    if self.trading_logger:
                        self.trading_logger.info("Stop loss triggered for short position")
                
                # Trailing stop
                elif unrealized_ticks > 10:  # Activate after 10 ticks profit
                    if not self.trailing_stop_active:
                        self.trailing_stop_active = True
                        self.trailing_stop_high = current_price
                    else:
                        # Update trailing stop low
                        if current_price < self.trailing_stop_high:
                            self.trailing_stop_high = current_price
                        # Check if we've pulled back too much
                        elif (current_price - self.trailing_stop_high) / self.tick_size > self.max_drawdown_ticks:
                            self.buy(state)
                            if self.trading_logger:
                                self.trading_logger.info("Trailing stop triggered for short position")
        '''
    
    # Insert risk management before reward calculation
    pattern = r'(# Calculate reward\s+reward = self\.get_reward\(state\))'
    content = re.sub(pattern, step_enhancement + '\n\n        \\1', content)
    
    # 3. Reset trailing stop when opening new position
    buy_reset = '''
        # Reset risk management for new position
        self.trailing_stop_active = False
        self.trailing_stop_high = None'''
    
    # Add to buy method after position is opened
    pattern = r'(self\._entry_step = self\.current_index.*?# Track entry step for holding period)'
    content = re.sub(pattern, r'\1' + buy_reset, content, flags=re.DOTALL)
    
    # Add to sell method after position is opened
    content = re.sub(pattern, r'\1' + buy_reset, content, flags=re.DOTALL)
    
    # 4. Simplify reward to focus on actual P&L with risk-adjusted bonuses
    reward_method = '''def get_reward(self, state: TimeSeriesState) -> float:
        """Risk-adjusted reward system focusing on consistent profits"""
        
        # Only reward closed positions (actual P&L)
        if self.current_position == 0 and self.last_position != 0:
            # Calculate actual P&L
            if self.last_position == 1:  # Closed long
                gross_pnl = ((self._last_closed_exit_price - self._last_closed_entry_price) / self.tick_size) * self.value_per_tick
            else:  # Closed short
                gross_pnl = ((self._last_closed_entry_price - self._last_closed_exit_price) / self.tick_size) * self.value_per_tick
            
            # Subtract costs
            net_pnl = gross_pnl - (2 * self.execution_cost_per_order)
            
            # Risk-adjusted reward multiplier
            risk_reward_ratio = abs(net_pnl) / (self.stop_loss_ticks * self.value_per_tick)
            
            if net_pnl > 0:
                # Reward good risk/reward trades more
                if risk_reward_ratio > 2.0:  # Won more than 2x risk
                    reward = net_pnl * 1.5
                elif risk_reward_ratio > 1.5:  # Won more than 1.5x risk
                    reward = net_pnl * 1.25
                else:
                    reward = net_pnl
                
                # Bonus for hitting take profit (disciplined exit)
                if abs(gross_pnl / self.value_per_tick) >= self.take_profit_ticks - 2:
                    reward += 20  # $20 bonus for disciplined profit taking
            else:
                # Smaller penalty if stopped out properly
                if abs(gross_pnl / self.value_per_tick) <= self.stop_loss_ticks + 2:
                    reward = net_pnl * 0.8  # 20% reduction for proper stop loss
                else:
                    reward = net_pnl  # Full loss if let it run beyond stop
            
            return reward
        
        # Small penalty for holding losing positions too long
        if self.current_position != 0 and self.entry_price is not None:
            current_price = state.price
            
            if self.current_position == 1:
                unrealized_ticks = (current_price - self.entry_price) / self.tick_size
            else:
                unrealized_ticks = (self.entry_price - current_price) / self.tick_size
            
            # Penalty for letting losses run
            if unrealized_ticks < -self.stop_loss_ticks:
                return -1.0  # Should have stopped out
            
            # Small reward for letting profits run
            elif unrealized_ticks > self.take_profit_ticks:
                return 0.5  # Exploring beyond take profit
            
            return 0.0
        
        # Neutral when flat
        return 0.0'''
    
    # Replace get_reward method
    reward_pattern = r'def get_reward\(self, state: TimeSeriesState\) -> float:.*?(?=\n    def|\n\nclass|\Z)'
    content = re.sub(reward_pattern, reward_method, content, flags=re.DOTALL)
    
    # Write the fixed file
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("\n✅ Fundamental fix applied!")
    print("\nKey improvements:")
    print("1. ✓ Automatic stop loss at 10 ticks ($25) to limit losses")
    print("2. ✓ Take profit at 20 ticks ($50) for 2:1 risk/reward")
    print("3. ✓ Trailing stop after 10 ticks profit to protect gains")
    print("4. ✓ Risk-adjusted rewards favor good risk/reward trades")
    print("5. ✓ Bonuses for disciplined exits at targets")
    print("\nExpected results:")
    print("- Maximum loss per trade: $25 (vs $50+ before)")
    print("- Target profit per trade: $50 (2:1 risk/reward)")
    print("- Win rate needed for profitability: >33% (due to 2:1 R/R)")
    print("- Should achieve 50%+ episode profitability")
    print("\nThe agent will learn to:")
    print("- Take profits at reasonable levels")
    print("- Cut losses quickly")
    print("- Let winners run with trailing stops")
    print("- Focus on risk/reward, not just activity")

if __name__ == "__main__":
    apply_fundamental_profitability_fix()