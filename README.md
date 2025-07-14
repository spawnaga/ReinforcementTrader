# AI Trading System API

A GPU-accelerated trading system using reinforcement learning for NQ futures trading.

## Quick Start

```bash
# 1. Install dependencies (if not already installed)
pip install -r requirements.txt

# 2. Set up environment variables
export DATABASE_URL="postgresql://user:pass@localhost/trading_db"

# 3. Run the application
python run.py
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

## API Usage Examples

### Start Training
```bash
curl -X POST http://localhost:5000/api/start_training \
  -H "Content-Type: application/json" \
  -d '{"name": "My Training Session", "algorithm": "ANE_PPO"}'
```

### Get Sessions
```bash
curl http://localhost:5000/api/sessions
```

### Get Trades
```bash
curl http://localhost:5000/api/trades?session_id=1
```

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