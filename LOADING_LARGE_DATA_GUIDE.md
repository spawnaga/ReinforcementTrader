# Guide for Loading Your 6 Million Row NQ Futures Data

## Quick Start

### Step 1: Test Data Loading
First, make sure your data file is in the `./data` folder, then run:

```bash
python test_load_data.py
```

This will:
- Find your large .txt file automatically
- Load a 1000-row sample to verify format
- Show data statistics
- Ask if you want to load the full file

### Step 2: Load Full Data to Database (Optional)
If you want to save to PostgreSQL for faster subsequent loads:

```bash
# Load and save to database
python load_large_nq_data.py --file ./data/your_nq_data.txt --save-db

# Or just test with a sample
python load_large_nq_data.py --file ./data/your_nq_data.txt --sample 100000
```

### Step 3: Prepare Data for Training
This adds technical indicators and creates train/test splits:

```bash
# Full preparation workflow
python prepare_training_data.py --file ./data/your_nq_data.txt

# Test with smaller dataset
python prepare_training_data.py --file ./data/your_nq_data.txt --max-states 10000 --test-training
```

## Your Data Format
The scripts expect this format (no headers):
```
2008-01-02 06:00:00,3602.50,3603.75,3601.75,3603.25,184
timestamp,open,high,low,close,volume
```

## Performance Tips for 6 Million Rows

1. **Use Caching**: The GPU data loader automatically caches processed data in `./data_cache` for faster subsequent loads

2. **Chunk Loading**: For 302MB file, data is loaded in 100k row chunks to avoid memory issues

3. **Database Storage**: After first load, save to PostgreSQL for much faster access:
   - First load from .txt: ~2-5 minutes
   - Subsequent loads from DB: ~10-30 seconds

4. **Training Optimization**: 
   - Start with smaller samples (100k-500k rows) to test algorithms
   - Use `--max-states` parameter to limit training data size
   - Full 6M rows may require 16-32GB RAM

## Using with Trading CLI

Once data is loaded, use the trading CLI:

```bash
# Train with your data
python trading_cli.py train \
    --algorithm ane-ppo \
    --ticker NQ \
    --data-source file \
    --data-file ./data/your_nq_data.txt \
    --episodes 1000 \
    --use-gpu

# Or if saved to database
python trading_cli.py train \
    --algorithm ane-ppo \
    --ticker NQ \
    --data-source database \
    --episodes 1000 \
    --use-gpu
```

## Troubleshooting

1. **Out of Memory**: Reduce chunk size or use `--sample` parameter
2. **Slow Loading**: Enable GPU acceleration if available (requires CUDA)
3. **Cache Issues**: Delete `./data_cache` folder to force reload

## Expected Processing Times

- Initial load (6M rows): 2-5 minutes
- Adding indicators: 1-2 minutes  
- Creating training states: 2-3 minutes
- Total first run: ~10 minutes

Subsequent runs use cache: ~30 seconds total