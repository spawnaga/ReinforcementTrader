# AI Trading System Dependencies

This document lists all the dependencies required for the AI Trading System.

## Core Dependencies

### Web Framework
- **flask** >= 3.1.1 - Web application framework
- **flask-sqlalchemy** >= 3.1.1 - Database ORM integration
- **flask-socketio** >= 5.5.1 - WebSocket support
- **gunicorn** >= 23.0.0 - WSGI HTTP Server
- **werkzeug** >= 3.1.3 - WSGI utility library

### Database
- **psycopg2-binary** >= 2.9.10 - PostgreSQL adapter
- **sqlalchemy** >= 2.0.41 - SQL toolkit and ORM

### Machine Learning
- **torch** >= 2.7.1 - PyTorch deep learning framework
- **numpy** >= 2.3.1 - Numerical computing
- **pandas** >= 2.3.1 - Data analysis and manipulation
- **scikit-learn** >= 1.7.0 - Machine learning library
- **scipy** >= 1.16.0 - Scientific computing
- **gym** >= 0.26.2 - Reinforcement learning environments

### Data Sources
- **yfinance** >= 0.2.65 - Yahoo Finance data
- **ib-insync** >= 0.9.86 - Interactive Brokers API

### Real-time Communication
- **python-socketio** >= 5.13.0 - Socket.IO server
- **eventlet** >= 0.40.1 - Concurrent networking library
- **websocket-client** >= 1.8.0 - WebSocket client

### Utilities
- **python-dotenv** >= 1.1.1 - Environment variable management
- **pytz** >= 2025.2 - Timezone support
- **psutil** >= 7.0.0 - System monitoring
- **matplotlib** >= 3.10.3 - Plotting library
- **email-validator** >= 2.2.0 - Email validation

## Installation

To install all dependencies, run:

```bash
pip install flask>=3.1.1 flask-sqlalchemy>=3.1.1 flask-socketio>=5.5.1 gunicorn>=23.0.0 werkzeug>=3.1.3 psycopg2-binary>=2.9.10 sqlalchemy>=2.0.41 torch>=2.7.1 numpy>=2.3.1 pandas>=2.3.1 scikit-learn>=1.7.0 scipy>=1.16.0 gym>=0.26.2 yfinance>=0.2.65 ib-insync>=0.9.86 python-socketio>=5.13.0 eventlet>=0.40.1 websocket-client>=1.8.0 python-dotenv>=1.1.1 pytz>=2025.2 psutil>=7.0.0 matplotlib>=3.10.3 email-validator>=2.2.0
```

Or use the package manager in your environment.

## Optional Dependencies

### Technical Analysis
- **ta** >= 0.11.0 - Technical analysis library (optional - we have manual implementations)

### Development Tools
- **pytest** >= 7.0.0 - Testing framework
- **black** >= 22.0.0 - Code formatter
- **flake8** >= 4.0.0 - Code linter
- **mypy** >= 0.950 - Type checker

## GPU Support

For GPU support with CUDA, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Notes

1. **TA-Lib**: The system includes manual implementations of all technical indicators, so TA-Lib is not required.
2. **Python Version**: Requires Python 3.11 or higher
3. **PostgreSQL**: Must be installed separately and accessible
4. **CUDA**: For GPU acceleration, NVIDIA CUDA toolkit must be installed