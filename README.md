# Revolutionary AI Trading System - Backend API

## Overview

A high-performance, GPU-accelerated trading backend that leverages advanced machine learning technologies for intelligent market analysis and adaptive neural trading strategies. This is a pure API backend designed to be integrated with any frontend framework.

## Core Features

- **RESTful API**: Clean, well-documented API endpoints for all trading operations
- **WebSocket Support**: Real-time data streaming and performance metrics
- **GPU Acceleration**: CUDA-enabled PyTorch for high-speed neural network training
- **Advanced Algorithms**: ANE-PPO (Attention Network Enhanced Proximal Policy Optimization)
- **PostgreSQL Database**: High-concurrency database configuration
- **Real-time Trading**: Integration with Interactive Brokers API
- **Genetic Optimization**: Hyperparameter tuning using genetic algorithms

## Technology Stack

### Backend
- **Flask**: RESTful API framework
- **SQLAlchemy**: ORM with PostgreSQL
- **Flask-SocketIO**: WebSocket support for real-time updates
- **Gunicorn**: Production-ready WSGI server

### Machine Learning
- **PyTorch**: Deep learning framework with CUDA support
- **OpenAI Gym**: Custom futures trading environment
- **NumPy/Pandas**: Data manipulation and analysis
- **Technical Analysis (TA)**: Advanced technical indicators

### Trading Infrastructure
- **Interactive Brokers API**: Live market data and order execution
- **Yahoo Finance**: Historical data backup source
- **Custom Risk Manager**: Position sizing and risk controls

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL database
- CUDA-capable GPU (optional, but recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd trading-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export DATABASE_URL="postgresql://user:password@localhost/trading_db"
export SESSION_SECRET="your-secret-key"
```

4. Initialize the database:
```bash
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

5. Start the server:
```bash
gunicorn --bind 0.0.0.0:5000 --reload main:app
```

## API Documentation

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

### Key Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/sessions` - List all trading sessions
- `POST /api/start_training` - Start AI training
- `GET /api/trades` - Retrieve trading history
- `GET /api/status` - System status and metrics

### WebSocket Events

Connect to `/socket.io/` for real-time updates:
- `performance_metrics` - CPU/Memory/GPU usage
- `training_update` - Training progress
- `new_trade` - Trade notifications

## Architecture

### Trading Engine
- **ANE-PPO Algorithm**: Advanced reinforcement learning with transformer attention
- **Multi-Scale Feature Extraction**: Processes market data at multiple timeframes
- **Risk Management**: Integrated position sizing and drawdown controls
- **Backtesting**: Historical performance validation

### Data Pipeline
- **GPU-Accelerated Loading**: Fast data processing with CUDA
- **Feature Engineering**: 50+ technical indicators
- **Market Regime Detection**: Transformer-based market state classification
- **Real-time Updates**: WebSocket streaming of market data

## Configuration

Key configuration options in `config.py`:

```python
# Machine Learning
ML_LEARNING_RATE = 3e-4
ML_GAMMA = 0.99
ML_N_STEPS = 2048
ML_BATCH_SIZE = 64

# Trading
MAX_POSITION_SIZE = 10
MAX_DAILY_LOSS = 1000.0
MAX_DRAWDOWN = 0.15

# Database
SQLALCHEMY_POOL_SIZE = 30
SQLALCHEMY_MAX_OVERFLOW = 20
```

## Integration Examples

### Python Client
```python
import requests

# Start training session
response = requests.post('http://localhost:5000/api/start_training', json={
    'session_name': 'My Trading Bot',
    'algorithm_type': 'ANE_PPO',
    'total_episodes': 1000
})
session_id = response.json()['session_id']

# Check status
status = requests.get('http://localhost:5000/api/status').json()
print(f"Active sessions: {status['active_sessions']}")
```

### JavaScript/WebSocket
```javascript
const socket = io('http://localhost:5000');

socket.on('performance_metrics', (data) => {
    console.log(`CPU: ${data.cpu_usage}%, Memory: ${data.memory_usage}%`);
});

socket.on('new_trade', (trade) => {
    console.log(`New ${trade.position_type} trade at ${trade.entry_price}`);
});
```

## Production Deployment

1. Use environment variables for all sensitive configuration
2. Enable HTTPS with proper SSL certificates
3. Implement authentication and rate limiting
4. Use a reverse proxy (Nginx) for load balancing
5. Set up monitoring and logging (e.g., Prometheus, Grafana)
6. Configure CORS for your specific frontend domain

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is proprietary software. All rights reserved.

## Support

For issues and questions, please open an issue in the repository.