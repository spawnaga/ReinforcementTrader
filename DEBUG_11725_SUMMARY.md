# 11,725 Reward Bug Investigation Summary

## Bug Description
After episode 50, the agent stops trading (0 trades) but shows a consistent reward of ~11,725-11,745.

## Key Findings

### 1. Curriculum Learning Transition (Episode 51)
- Episode length: 300 → 200 steps (33% reduction)
- Trade limit: 10 → 7 (30% reduction)
- Min holding period: 5 → 8 steps (60% increase)
- Cost per trade: $2.50 → $5.00 (100% increase)

### 2. Suspicious Value Analysis
- 11,725 = 234,500 / 20 (where 20 is NQ contract multiplier)
- Consistent across episodes with 0 trades
- Expected penalty for 150-200 steps of no trading: -11.25 to -15 (not +11,725!)

### 3. Verified Working Correctly
- ✓ train_standalone.py accumulates rewards correctly
- ✓ Environment step() returns reward correctly
- ✓ calculate_reward() in utils.py returns net_profit correctly
- ✓ Curriculum parameters ARE being passed to environment

### 4. Potential Sources (Still Investigating)
- [ ] Observation values leaking into rewards
- [ ] Portfolio value being used instead of step reward
- [ ] Initialization bug when episode_number changes
- [ ] Reward accumulation from previous episodes

## Files to Test on Your Local Machine

### 1. Enhanced train_standalone.py
Already includes debugging for episodes 45-55 and tracks 11,725 values.

### 2. Test Scripts Created
- `debug_11725_reward.py` - Systematic investigation of the value
- `test_reward_accumulation.py` - Direct test of episode 51 rewards
- `check_observation_values.py` - Check if 11,725 is in observations

### 3. Key Code Sections to Monitor

In train_standalone.py (lines 420-430):
```python
# Debug for mysterious 11,725 reward
if 45 <= episode <= 55 and abs(reward) > 10:
    loggers['algorithm'].warning(
        f"Episode {episode} Step {step}: Large reward {reward:.2f}"
    )
```

In futures_env_realistic.py (lines 739-746):
```python
# Episode length changes with curriculum
if self.episode_number < 50:
    self.limit = min(len(self.states), 300)
elif self.episode_number < 150:
    self.limit = min(len(self.states), 200)
```

## Testing Steps for Your Local Machine

1. Run training with your 5M row dataset:
   ```bash
   python train_standalone.py --data-path your_data.csv --episodes 100 --num-gpus 4
   ```

2. Monitor the logs, especially:
   - `logs/algorithm.log` - Look for "Large reward" warnings
   - `logs/rewards.log` - Track individual step rewards
   - `logs/trading.log` - Verify trades stop after episode 50

3. Check for the 11,725 value appearing:
   - First occurrence (which episode?)
   - Is it exactly 11,725 or slightly different?
   - Does it appear in observation values or just rewards?

4. Run the debug script with your data:
   ```bash
   python debug_11725_reward.py
   ```

## Hypothesis to Test

The 11,725 might be:
1. **Price data** - NQ futures price around 14,880 getting into rewards
2. **Portfolio value** - Total account value instead of step reward
3. **Observation contamination** - Large values from state representation
4. **Accumulation bug** - Previous episode's total carrying over

## What We Need From Your Test

1. Exact value when it first appears
2. Whether it's in rewards or observations
3. If it correlates with any market data values
4. Console output from the debug scripts

Your local testing with real data will help us pinpoint the exact source of this bug.