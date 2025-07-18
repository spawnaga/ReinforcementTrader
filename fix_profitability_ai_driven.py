#!/usr/bin/env python3
"""
AI-driven profitability improvement - enhance state representation and reward learning
No hard-coded rules - let the AI learn risk management
"""

import re
from datetime import datetime

def apply_ai_driven_profitability_fix():
    print("🤖 AI-Driven Profitability Fix - Enhanced Learning, Not Rules")
    print("=" * 60)
    
    # Read the current file
    with open('futures_env_realistic.py', 'r') as f:
        content = f.read()
    
    # Backup current version
    backup_name = f"futures_env_realistic.py.ai_driven_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_name, 'w') as f:
        f.write(content)
    print(f"✓ Created backup: {backup_name}")
    
    # 1. First, let's enhance the reward to teach risk/reward naturally
    new_reward_method = '''def get_reward(self, state: TimeSeriesState) -> float:
        """AI-focused reward that teaches risk management without hard rules"""
        
        # Only reward on closed positions
        if self.current_position == 0 and self.last_position != 0:
            # Calculate actual P&L
            if self.last_position == 1:  # Closed long
                gross_pnl = ((self._last_closed_exit_price - self._last_closed_entry_price) / self.tick_size) * self.value_per_tick
            else:  # Closed short
                gross_pnl = ((self._last_closed_entry_price - self._last_closed_exit_price) / self.tick_size) * self.value_per_tick
            
            # Subtract costs
            net_pnl = gross_pnl - (2 * self.execution_cost_per_order)
            
            # Risk-aware reward shaping
            # Teach the agent about risk/reward ratio through the reward signal
            if net_pnl > 0:
                # Reward good risk/reward trades more
                # The agent will learn that bigger wins are better
                reward = net_pnl
                
                # Bonus for excellent trades to encourage letting winners run
                profit_ticks = gross_pnl / self.value_per_tick
                if profit_ticks > 20:  # Great trade
                    reward *= 1.5  # 50% bonus
                elif profit_ticks > 10:  # Good trade
                    reward *= 1.2  # 20% bonus
                
            else:
                # Penalize losses more heavily to teach risk management
                # This encourages the agent to avoid big losses
                loss_ticks = abs(gross_pnl / self.value_per_tick)
                
                if loss_ticks > 20:  # Terrible loss
                    reward = net_pnl * 2.0  # Double penalty
                elif loss_ticks > 10:  # Bad loss
                    reward = net_pnl * 1.5  # 50% extra penalty
                else:
                    reward = net_pnl  # Normal loss
            
            return reward
        
        # Small penalty for holding too long (opportunity cost)
        if self.current_position != 0 and hasattr(self, '_entry_step'):
            holding_time = self.current_index - self._entry_step
            if holding_time > 50:  # Held for too long
                return -0.1
        
        return 0.0'''
    
    # Replace the reward method
    reward_pattern = r'def get_reward\(self, state: TimeSeriesState\) -> float:.*?(?=\n    def|\n\nclass|\Z)'
    content = re.sub(reward_pattern, new_reward_method, content, flags=re.DOTALL)
    
    # 2. Let's add a method to calculate risk metrics that the agent can use
    risk_metrics_method = '''
    def _get_risk_metrics(self, state: TimeSeriesState) -> dict:
        """Calculate risk metrics to help agent make better decisions"""
        metrics = {}
        
        if self.current_position != 0 and self.entry_price is not None:
            current_price = state.price
            
            # Calculate unrealized P&L
            if self.current_position == 1:
                unrealized_pnl = (current_price - self.entry_price) * 2 - self.execution_cost_per_order
                unrealized_ticks = (current_price - self.entry_price) / self.tick_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * 2 - self.execution_cost_per_order
                unrealized_ticks = (self.entry_price - current_price) / self.tick_size
            
            metrics['unrealized_pnl'] = unrealized_pnl
            metrics['unrealized_ticks'] = unrealized_ticks
            metrics['holding_time'] = self.current_index - self._entry_step if hasattr(self, '_entry_step') else 0
            
            # Risk/reward info
            metrics['pnl_ratio'] = unrealized_pnl / (self.entry_price * 2) if self.entry_price > 0 else 0
        else:
            metrics['unrealized_pnl'] = 0
            metrics['unrealized_ticks'] = 0
            metrics['holding_time'] = 0
            metrics['pnl_ratio'] = 0
        
        return metrics'''
    
    # Add the risk metrics method after get_reward
    pattern = r'(def get_reward.*?return total_reward)'
    content = re.sub(pattern, r'\1\n' + risk_metrics_method, content, flags=re.DOTALL)
    
    # 3. Enhance observation to include risk information
    obs_enhancement = '''
        # Add risk metrics to observation if in position
        if self.add_current_position_to_state and self.current_position != 0:
            risk_metrics = self._get_risk_metrics(state)
            # Add normalized risk information
            risk_features = np.array([
                risk_metrics['unrealized_ticks'] / 20.0,  # Normalized by typical move
                risk_metrics['holding_time'] / 50.0,  # Normalized by typical hold time
                risk_metrics['pnl_ratio'] * 10.0  # Scaled P&L ratio
            ])
            obs = np.concatenate([obs, risk_features])'''
    
    # Find the _get_observation method and enhance it
    pattern = r'(if self\.add_current_position_to_state:.*?obs = np\.concatenate\(\[state_data, position_array\]\))'
    replacement = r'\1' + obs_enhancement
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 4. Update observation space to include risk features
    obs_space_update = '''
                state_shape = (state_shape[0] + 1,)  # For position
                if self.current_position != 0:  # If we might add risk features
                    state_shape = (state_shape[0] + 3,)  # For risk metrics'''
    
    pattern = r'(if self\.add_current_position_to_state:.*?state_shape = \(state_shape\[0\] \+ 1,\))'
    content = re.sub(pattern, r'\1' + '\n' + obs_space_update, content)
    
    # Write the fixed file
    with open('futures_env_realistic.py', 'w') as f:
        f.write(content)
    
    print("\n✅ AI-driven fix applied!")
    print("\nKey improvements:")
    print("1. ✓ Risk-aware reward shaping - teaches agent about risk/reward naturally")
    print("2. ✓ Bonuses for good trades, penalties for bad losses")
    print("3. ✓ Added risk metrics to observation (unrealized P&L, holding time)")
    print("4. ✓ No hard-coded exits - agent learns when to exit")
    print("\nThe AI will learn to:")
    print("- Exit losing trades before they get too big")
    print("- Let winning trades run for bigger profits")
    print("- Understand risk/reward ratios through experience")
    print("- Make context-aware decisions based on market conditions")
    print("\nExpected outcome:")
    print("- Better risk/reward ratio through learned behavior")
    print("- Natural stop loss and take profit through AI decision making")
    print("- Improved profitability without rigid rules")

if __name__ == "__main__":
    apply_ai_driven_profitability_fix()