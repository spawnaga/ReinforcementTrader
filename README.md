# AI Trading System - Advanced Reinforcement Learning Platform

A comprehensive, GPU-accelerated trading system with reinforcement learning algorithms, supporting multiple futures contracts with full control over hardware, algorithms, and training parameters.

## 🚀 Key Features

- **Multi-GPU Support**: Utilize up to 4 GPUs with automatic detection and configuration
- **Advanced Algorithms**: ANE-PPO, Genetic Optimization, Q-Learning, Transformer Attention
- **Multiple Futures**: Support for NQ, ES, CL, GC and other futures contracts
- **Command-Line Interface**: Complete control through intuitive CLI commands
- **Technical Indicators**: All major indicators implemented (no TA-Lib dependency)
- **Real-time API**: RESTful API with WebSocket support for live updates
- **Risk Management**: Built-in stop-loss, take-profit, and position limits

## System Requirements

- Python 3.11+
- PostgreSQL database
- CUDA-capable GPU(s) (optional, CPU mode available)
- 16GB+ RAM recommended

## 📦 Dependencies

All project dependencies are documented in `DEPENDENCIES.md`. Key requirements include:
- Flask, Flask-SQLAlchemy, Flask-SocketIO for web framework
- PyTorch for deep learning (with optional CUDA support)
- NumPy, Pandas, Scikit-learn for data processing
- psycopg2-binary for PostgreSQL
- See `DEPENDENCIES.md` for the complete list with versions

**Note**: The `requirements.txt` file is protected in this environment. Refer to `DEPENDENCIES.md` for manual installation instructions.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
export DATABASE_URL="postgresql://user:pass@localhost/trading_db"

# 3. Run the API server
python run.py

# 4. Start training with CLI
python trading_cli.py --device gpu --ticker NQ --episodes 1000 --train
```

## 🎮 Command-Line Interface (CLI)

The `trading_cli.py` tool provides comprehensive control over all aspects of the trading system.

### GPU Configuration
```bash
# Use specific number of GPUs
python trading_cli.py --num-gpus 4 --ticker NQ --episodes 5000

# Use specific GPU IDs
python trading_cli.py --gpu-ids 0 1 2 3 --ticker ES --episodes 10000

# Force CPU mode
python trading_cli.py --device cpu --ticker NQ --episodes 1000
```

### Training Control
```bash
# Set training episodes
python trading_cli.py --episodes 5000        # Train for 5,000 episodes

# Configure training loops and steps
python trading_cli.py --training-loops 5 --epochs-per-loop 10 --max-steps 300

# Full training configuration
python trading_cli.py --episodes 10000 --batch-size 128 --learning-rate 0.0001
```

### Algorithm Selection
```bash
# ANE-PPO (default)
python trading_cli.py --algorithm ane_ppo --ticker NQ

# Genetic Optimization
python trading_cli.py --algorithm genetic --ticker ES

# Q-Learning
python trading_cli.py --algorithm q_learning --ticker CL

# With Transformer Attention
python trading_cli.py --algorithm ane_ppo --use-transformer --ticker GC
```

### Data Configuration
```bash
# Use percentage of data
python trading_cli.py --data-range percentage --percentage 50

# Use specific date range
python trading_cli.py --data-range daterange --start-date 2024-01-01 --end-date 2024-12-31

# Use time period
python trading_cli.py --data-range timeperiod --period 6months

# Use CSV file
python trading_cli.py --data-source csv --csv-path /path/to/data.csv
```

### Technical Indicators
```bash
# Use specific indicators
python trading_cli.py --indicators rsi macd bollinger stochastic

# Use all indicators
python trading_cli.py --indicators all

# Available indicators:
# - RSI (Relative Strength Index)
# - MACD (Moving Average Convergence Divergence)
# - Bollinger Bands
# - Stochastic Oscillator
# - Williams %R
# - ATR (Average True Range)
# - SMA/EMA (Simple/Exponential Moving Averages)
```

### Risk Management
```bash
# Set risk parameters
python trading_cli.py --max-position 5 --stop-loss 100 --take-profit 200

# Configure trading constraints
python trading_cli.py --max-trades 10 --min-holding 20 --slippage 3
```

## Core Components

### 1. **app.py** - Flask Application
- Initializes Flask app with PostgreSQL database
- Configures SocketIO for real-time updates
- Sets up database models

### 2. **routes.py** - API Endpoints
- `/health` - System health check
- `/api/sessions` - Trading session management
- `/api/start_training` - Start ML training
- `/api/stop_training` - Stop training
- `/api/trades` - Trade history
- `/api/market_data` - Market data access

### 3. **trading_engine.py** - Trading Logic
- ANE-PPO algorithm implementation
- Multi-GPU training support
- Real-time trade execution
- Session management

### 4. **futures_env_realistic.py** - Trading Environment
- Realistic trading constraints
- Anti-exploitation measures:
  - Minimum holding periods (10 steps)
  - Maximum 5 trades per episode
  - Transaction costs ($5-10 per trade)
  - Slippage simulation (0-2 ticks)

### 5. **data_manager.py** - Data Processing
- Loads NQ futures data
- Technical indicator calculation
- GPU-accelerated data processing

## 📊 Supported Futures Contracts

- **NQ** - NASDAQ-100 E-mini Futures
- **ES** - S&P 500 E-mini Futures  
- **CL** - Crude Oil Futures
- **GC** - Gold Futures
- **SI** - Silver Futures
- **ZB** - 30-Year US Treasury Bond Futures
- **6E** - Euro FX Futures
- **ZC** - Corn Futures

## 🧪 Complete Examples

### Example 1: Multi-GPU Training with NQ Futures
```bash
python trading_cli.py \
  --num-gpus 4 \
  --ticker NQ \
  --episodes 5000 \
  --algorithm ane_ppo \
  --indicators rsi macd bollinger \
  --batch-size 128 \
  --learning-rate 0.0001 \
  --max-trades 10 \
  --stop-loss 50 \
  --train
```

### Example 2: CPU Training with Custom Data
```bash
python trading_cli.py \
  --device cpu \
  --ticker ES \
  --data-source csv \
  --csv-path ./data/ES_2024.csv \
  --episodes 1000 \
  --algorithm genetic \
  --training-loops 5 \
  --train
```

### Example 3: Time Period Training
```bash
python trading_cli.py \
  --gpu-ids 0 1 \
  --ticker CL \
  --data-range timeperiod \
  --period 1year \
  --algorithm q_learning \
  --indicators all \
  --episodes 3000 \
  --train
```

## 🛠️ API Usage

The system provides a RESTful API for integration with other applications.

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/api/sessions` | GET | List all training sessions |
| `/api/sessions/<id>` | GET | Get specific session details |
| `/api/start_training` | POST | Start new training session |
| `/api/stop_training` | POST | Stop active training session |
| `/api/trades` | GET | Get trade history |
| `/api/recent_trades` | GET | Get recent trades |
| `/api/market_data` | GET | Retrieve market data |
| `/api/algorithm_configs` | GET | Get algorithm configurations |

### WebSocket Events

The system broadcasts real-time updates via WebSocket:

- `session_update` - Training session updates
- `trade_update` - New trade notifications
- `performance_metrics` - Real-time performance data
- `training_metrics` - Training progress and metrics

### API Examples

#### Start Training
```bash
curl -X POST http://localhost:5000/api/start_training \
  -H "Content-Type: application/json" \
  -d '{
    "name": "NQ Training Session",
    "algorithm": "ANE_PPO",
    "parameters": {
      "episodes": 1000,
      "device": "gpu",
      "ticker": "NQ"
    }
  }'
```

#### Get Sessions
```bash
curl http://localhost:5000/api/sessions
```

#### Get Recent Trades
```bash
curl http://localhost:5000/api/recent_trades?limit=10
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```bash
DATABASE_URL=postgresql://username:password@localhost/trading_db
SESSION_SECRET=your-secret-key-here
ENVIRONMENT=development
```

### Database Setup

The system uses PostgreSQL with automatic table creation. Ensure PostgreSQL is running and accessible.

```bash
# Create database
createdb trading_db

# The application will automatically create tables on startup
```

## 📁 Project Structure

```
.
├── trading_cli.py          # Command-line interface
├── app.py                  # Flask application with extensions
├── run.py                  # Application runner
├── config.py               # Unified configuration (app + trading)
├── routes.py               # API endpoints
├── models.py               # Database models
├── trading_engine.py       # Core trading logic
├── data_manager.py         # Data processing
├── risk_manager.py         # Risk management
├── technical_indicators.py # Technical analysis
├── futures_contracts.py    # Futures specifications
├── trading_logger.py       # Trading activity logger
├── websocket_handler.py    # Real-time WebSocket events
├── rl_algorithms/          # Reinforcement learning algorithms
│   ├── ane_ppo.py         # ANE-PPO implementation
│   ├── dqn.py             # Deep Q-Network
│   └── genetic.py         # Genetic optimization
├── gym_futures/            # Trading environments
│   └── envs/
│       └── futures_env_realistic.py
├── data/                   # Market data storage
├── models/                 # Trained model storage
├── logs/                   # Training logs
├── db_utils.py            # Database utilities
├── gpu_data_loader.py     # GPU-accelerated data loading
├── ib_integration.py      # Interactive Brokers integration
├── run_training.py        # Training coordinator
├── training_monitor.py    # Comprehensive monitoring tool
└── cleanup_local_database.py # Database maintenance
```

## 🚨 Troubleshooting

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode if needed
python trading_cli.py --device cpu
```

### Database Connection Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -U username -d trading_db -c "SELECT 1"
```

### Memory Issues
```bash
# Reduce batch size
python trading_cli.py --batch-size 16

# Limit data percentage
python trading_cli.py --data-range percentage --percentage 25
```

## 📈 Performance Tips

1. **Multi-GPU Training**: Use all available GPUs for faster training
   ```bash
   python trading_cli.py --num-gpus 4 --batch-size 256
   ```

2. **Optimal Batch Size**: Larger batch sizes improve GPU utilization
   ```bash
   python trading_cli.py --batch-size 128  # or 256 for better performance
   ```

3. **Data Caching**: The system automatically caches processed data for faster subsequent runs

4. **Algorithm Selection**: 
   - ANE-PPO: Best overall performance
   - Genetic: Good for hyperparameter optimization
   - DQN: Fast training, simpler architecture

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- OpenAI Gym for the environment structure
- Interactive Brokers for market data and execution
- PostgreSQL for reliable data storage

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting guide above
- Review the API documentation

## 📊 Recent Updates

### July 17, 2025 - Project Cleanup
- Reduced project from 44 to 29 files (34% reduction)
- Removed all duplicate monitoring scripts
- Consolidated configuration files
- Merged extensions into app.py
- Updated documentation with cleaned structure
- All functionality preserved with cleaner organization

---

**Version**: 1.0.1  
**Last Updated**: July 17, 2025

## WebSocket Events

Connect to receive real-time updates:
- `performance_update` - System performance metrics
- `trade_update` - New trade notifications
- `session_update` - Training progress

## Important Notes

1. **Database**: Requires PostgreSQL (no SQLite support)
2. **GPU**: Automatically uses CUDA if available
3. **Trading**: Implements realistic constraints to prevent exploitation
4. **API-Only**: This is a backend API service (no frontend)

## Project Structure

```
.
├── app.py                  # Flask app initialization
├── main.py                 # Gunicorn entry point
├── run.py                  # Simple run script
├── config.py               # Configuration settings
├── models.py               # Database models
├── routes.py               # API endpoints
├── trading_engine.py       # Core trading logic
├── data_manager.py         # Data processing
├── futures_env_realistic.py # Trading environment
├── rl_algorithms/          # ML algorithms
│   └── ane_ppo.py         # ANE-PPO implementation
└── gym_futures/            # Gym environment
    └── envs/
        └── futures_env.py  # Base environment
```