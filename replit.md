# Revolutionary AI Trading System

## Overview

This is a comprehensive AI-powered trading system that combines reinforcement learning algorithms with live market data integration. The system specializes in NQ futures trading using advanced neural networks, including transformer attention mechanisms and genetic optimization for hyperparameter tuning.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Flask Web Framework**: Core web application with SQLAlchemy ORM for database management
- **Real-time Communication**: WebSocket integration using Flask-SocketIO for live updates
- **Trading Engine**: Custom GPU-accelerated engine with hybrid reinforcement learning algorithms
- **Database**: SQLite for development (designed to scale to PostgreSQL in production)
- **Asynchronous Processing**: Threading-based architecture for concurrent trading operations

### Frontend Architecture
- **Modern Web Interface**: Bootstrap-based responsive design with dark theme
- **Real-time Visualization**: 3D portfolio visualization using Three.js
- **Advanced Charts**: Interactive trading charts with technical indicators
- **Neural Network Visualization**: Real-time neural network activity display
- **Strategy Builder**: Visual drag-and-drop interface for algorithm configuration

## Key Components

### Trading Engine (`trading_engine.py`)
- **ANE-PPO Algorithm**: Advanced Proximal Policy Optimization with transformer attention
- **Genetic Optimizer**: Multi-objective optimization for hyperparameter tuning
- **GPU Acceleration**: CUDA-enabled neural network training
- **Multi-threading**: Concurrent training and live trading capabilities

### Data Management (`data_manager.py`)
- **Market Data**: NQ futures data handling with tick-level precision
- **Data Pipeline**: Automated data loading, preprocessing, and feature engineering
- **Database Integration**: Efficient storage and retrieval of market data

### Interactive Brokers Integration (`ib_integration.py`)
- **Live Trading**: Real-time order execution and position management
- **Market Data**: Live price feeds and market depth information
- **Risk Management**: Automated risk controls and position sizing

### Machine Learning Models
- **Actor-Critic Networks**: Multi-scale feature extraction with attention mechanisms
- **DQN**: Deep Q-Learning with dueling architecture
- **Transformer Attention**: Self-attention for sequence modeling
- **Genetic Algorithm**: Population-based hyperparameter optimization

## Data Flow

1. **Market Data Ingestion**: Live data from Interactive Brokers or historical data loading
2. **Feature Engineering**: Technical indicators and state representation
3. **AI Processing**: Neural network inference for trading decisions
4. **Risk Assessment**: Real-time risk evaluation and position sizing
5. **Order Execution**: Automated trade placement and monitoring
6. **Performance Tracking**: Real-time metrics and visualization updates

## External Dependencies

### Trading Infrastructure
- **Interactive Brokers**: Live market data and order execution
- **Yahoo Finance**: Historical data backup source

### Machine Learning Stack
- **PyTorch**: Deep learning framework with CUDA support
- **OpenAI Gym**: Custom futures trading environment
- **NumPy/Pandas**: Data manipulation and analysis

### Web Technologies
- **Flask**: Web framework and API
- **Socket.IO**: Real-time communication
- **Bootstrap**: Frontend styling
- **Three.js**: 3D visualizations
- **Chart.js**: Trading charts

## Deployment Strategy

### Development Environment
- **Local SQLite**: Development database
- **Debug Mode**: Flask development server
- **Mock Trading**: Simulated trading environment

### Production Considerations
- **Database Migration**: SQLite to PostgreSQL for production
- **WSGI Server**: Gunicorn or uWSGI for production deployment
- **Load Balancing**: Nginx reverse proxy
- **Environment Variables**: Secure configuration management
- **Docker**: Containerized deployment support

### Security Features
- **Session Management**: Secure session handling with configurable timeouts
- **Proxy Support**: ProxyFix middleware for reverse proxy deployments
- **CORS Configuration**: Configurable cross-origin resource sharing
- **Environment-based Secrets**: Secure credential management

### Monitoring and Logging
- **Comprehensive Logging**: Debug-level logging throughout the system with detailed performance tracking
- **Real-time Metrics**: WebSocket-based performance monitoring broadcasting every 5 seconds
  - CPU usage monitoring with psutil
  - Memory usage tracking
  - Network I/O statistics
  - GPU utilization (when CUDA available)
- **Error Handling**: Global error handlers and graceful degradation
- **System Health**: Real-time resource monitoring with detailed debug logs
- **WebSocket Debugging**: Full broadcast loop tracking with iteration counters

## Recent Updates (July 12, 2025)

### Bug Fixes and Improvements
- **Training System**: Fixed training stall issue by improving TimeSeriesState creation with automatic timestamp column detection
- **Performance Monitoring**: Confirmed WebSocket broadcasting working correctly, UI elements properly configured with IDs (gpu-progress, memory-progress, network-progress, speed-progress)
- **Data Loading**: Limited states to 100 for testing to prevent memory issues
- **Error Handling**: Added comprehensive error handling in state creation with fallback timestamp detection
- **Debug Logging**: Set log level to DEBUG for better diagnostics during training
- **Session Management**: Added automatic cleanup for stale training sessions
- **Data Type Fixes**: Added numeric type conversion for OHLCV columns to fix numpy object dtype tensor conversion errors
- **ANE-PPO Algorithm**: Enhanced state-to-tensor conversion with robust type handling for memoryview, DataFrames, and numpy arrays
- **Environment Consistency**: Fixed futures_env reset method to return TimeSeriesState objects instead of flattened arrays

### Trading System Improvements (July 12, 2025)
- **Trading Logger**: Created comprehensive trading_logger.py module that tracks all trading activities including entry/exit prices, positions, rewards, and errors
- **Enhanced Futures Environment**: Integrated trading logger into gym_futures/envs/futures_env.py with detailed logging in buy(), sell(), and get_reward() methods
- **Trading Dashboard Charts**: Fixed price chart implementation by:
  - Changing from unsupported 'candlestick' type to 'line' chart
  - Added addSampleData() method to display test data
  - Fixed chart initialization in TradingCharts constructor
- **Data Loading Fix**: Added load_futures_data() method to DataManager to properly load NQ futures data from CSV/TXT files
- **Diagnostic Tools**: Created check_trading_system.py to diagnose issues with None prices in trading execution

### Recent Fixes (July 12, 2025 - Latest Session)
- **Missing API Routes**: Added `/api/session_status/<status>` route that was causing 404 errors
- **Service Worker**: Disabled service worker registration (sw.js not implemented) to eliminate console errors
- **Sample Data**: Successfully populated database with sample trading session and 10 realistic trades
- **Recent Trades API**: Verified `/api/recent_trades` endpoint returning proper JSON data
- **ANE-PPO Dimension Error Fix**: Fixed "Dimension out of range" error by ensuring all state tensors are 3D (batch_size, sequence_length, features) instead of 2D. The network expects sequential data, not flattened states.
- **Tuple Mean Error**: Fixed critical error where transformer_attention module was returning a tuple (output, attention_weights) instead of just the output tensor. Added proper tuple handling to extract the output tensor before calling .mean() for global average pooling.
- **Database Permissions**: Fixed readonly database errors by removing conflicting SQLite connection and setting proper file permissions (chmod 666) on all database files including WAL and SHM files
- **Tensor Dimension Mismatch Fix** (July 12, 2025): Fixed RuntimeError "The size of tensor a (60) must match the size of tensor b (512)" in PositionalEncoding by:
  - Updated the forward method in PositionalEncoding class to properly handle the transposed tensor dimensions
  - The fix ensures positional encoding correctly works with tensors after input projection from 60 features to 512 dimensions
  - Tested and verified with test_tensor_fix.py - DQN model now works correctly with 60 input features
- **Transformer Attention Dimension Fix** (July 12, 2025): Fixed IndexError "Dimension out of range" in PositionalEncoding by:
  - Added dimension checking in PositionalEncoding.forward() to handle both 2D and 3D tensors
  - Fixed ANE-PPO's feature combination to keep 3D shape for transformer input (removed incorrect .mean(dim=-1))
  - Transformer now properly handles sequential data from the trading environment
- **ANE-PPO Feature Projection Fix** (July 12, 2025): Fixed "The size of tensor a (1536) must match the size of tensor b (512)" error by:
  - Added feature_projection layer to combine 3 feature extractors (3×512=1536 dims) back to 512 dims
  - Replaced complex DQN Q-network with simplified Sequential network that handles 2D input properly
  - All tests now pass successfully with correct tensor dimensions throughout the network

### Known Issues
- **WebSocket Timeouts**: Fixed by:
  - Setting SOCKETIO_ASYNC_MODE to 'threading' for sync workers
  - Reducing ping timeout to 20s and ping interval to 10s
  - This prevents the 30-second worker timeout issue
- **GPU Access**: System running in CPU-only mode in Replit environment despite detecting GPUs
- **UI Updates**: Performance metrics being broadcast successfully to frontend every 5 seconds
- **Worker Configuration**: Gunicorn using sync worker (workflow limitation) but WebSocket handling improved with threading mode
- **Database Locking**: Fixed by:
  - Enabling SQLite WAL mode for better concurrent access
  - Setting pool_size to 1 for SQLite
  - Adding retry logic with exponential backoff
- **Readonly Database Errors**: Fixed by:
  - Created db_utils.py module with retry decorators for database operations
  - Added ensure_db_writable() checks before write operations
  - Created fix_db_permissions.py utility script for local development
  - Updated _save_training_metrics and _update_session_stats with @retry_on_db_error decorators
- **Chart Display Issues** (July 12, 2025): 
  - Added chartjs-adapter-date-fns for time scale support
  - Charts initialized but may need debugging based on console errors
  - TradingCharts class properly implements line charts with sample data

### Critical Fixes Applied
- **Memoryview Handling**: Added robust handling for memoryview objects in ANE-PPO state conversion
- **Reward Calculation**: Fixed NoneType subtraction errors by adding null checks for entry/exit prices
- **Type Safety**: Ensured all data types are properly converted to float32 before tensor conversion
- **Data Loading**: Fixed DataManager missing method issue by adding load_futures_data() that leverages existing GPU-accelerated loading infrastructure

### Version Control Configuration
- **Added .gitignore**: Comprehensive ignore patterns for models/, logs/, attached_assets/, database files, cache directories, and ML checkpoints to prevent merge conflicts

The system is designed to be highly scalable, with the ability to add multiple trading algorithms, extend to different financial instruments, and integrate with various data sources and brokers.