# Revolutionary AI Trading System API Documentation

## Overview

This is a RESTful API backend for the Revolutionary AI Trading System. The system provides endpoints for managing trading sessions, executing trades, and monitoring AI-driven trading algorithms.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, the API does not require authentication. In production, implement appropriate authentication mechanisms.

## Endpoints

### System Information

#### GET /
Returns basic API information and available endpoints.

**Response:**
```json
{
  "name": "Revolutionary AI Trading System API",
  "version": "2.0.0",
  "type": "Backend API",
  "documentation": "/api/docs",
  "health": "/health",
  "endpoints": {
    "sessions": "/api/sessions",
    "trades": "/api/trades",
    "training": "/api/start_training",
    "status": "/api/status"
  }
}
```

#### GET /health
Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-14T03:02:12.434583+00:00"
}
```

### Trading Sessions

#### GET /api/sessions
Retrieve all trading sessions.

**Response:**
```json
[
  {
    "id": 1,
    "name": "Training Session",
    "algorithm_type": "ANE_PPO",
    "status": "active",
    "start_time": "2025-07-14T02:00:00.000000",
    "end_time": null,
    "total_episodes": 1000,
    "current_episode": 150,
    "total_profit": 1250.50,
    "total_trades": 45,
    "win_rate": 0.65,
    "sharpe_ratio": 1.8,
    "max_drawdown": 0.12
  }
]
```

#### POST /api/sessions
Create a new trading session.

**Request Body:**
```json
{
  "session_name": "My Trading Session",
  "algorithm_type": "ANE_PPO",
  "total_episodes": 1000,
  "parameters": {
    "learning_rate": 0.0003,
    "gamma": 0.99
  }
}
```

#### GET /api/sessions/{session_id}
Get details of a specific session.

#### DELETE /api/sessions/{session_id}
Delete a trading session and all related data.

#### POST /api/sessions/{session_id}/start
Start a training session.

#### POST /api/sessions/{session_id}/pause
Pause a running session.

#### POST /api/sessions/{session_id}/stop
Stop a training session.

#### POST /api/sessions/{session_id}/reset
Reset a session, clearing all trades and metrics.

### Training Control

#### POST /api/start_training
Start a new training session with specified parameters.

**Request Body:**
```json
{
  "session_name": "Advanced Training",
  "algorithm_type": "ANE_PPO",
  "total_episodes": 5000,
  "parameters": {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "clip_range": 0.2,
    "n_steps": 2048
  }
}
```

#### POST /api/stop_training
Stop an active training session.

**Request Body:**
```json
{
  "session_id": 1
}
```

### Trading Data

#### GET /api/trades
Retrieve trades with optional filtering.

**Query Parameters:**
- `session_id` (optional): Filter by session
- `limit` (optional, default: 100): Maximum number of trades to return
- `offset` (optional, default: 0): Pagination offset

#### GET /api/trades/{trade_id}
Get details of a specific trade.

#### GET /api/recent_trades
Get the most recent trades across all sessions.

**Query Parameters:**
- `limit` (optional, default: 20): Number of trades to return

### Market Data

#### GET /api/market_data
Retrieve OHLCV market data.

**Query Parameters:**
- `symbol` (optional, default: 'NQ'): Trading symbol
- `timeframe` (optional, default: '1min'): Data timeframe
- `limit` (optional, default: 1000): Number of data points

### Training Metrics

#### GET /api/training_metrics/{session_id}
Get training metrics for a specific session.

**Response:**
```json
[
  {
    "episode": 100,
    "timestamp": "2025-07-14T02:30:00.000000",
    "reward": 125.50,
    "loss": 0.0012,
    "epsilon": 0.85,
    "learning_rate": 0.0003,
    "action_distribution": {
      "buy": 0.35,
      "sell": 0.25,
      "hold": 0.40
    }
  }
]
```

### Algorithm Configuration

#### GET /api/algorithm_configs
Get all available algorithm configurations.

#### POST /api/algorithm_configs
Create a new algorithm configuration.

**Request Body:**
```json
{
  "name": "Conservative ANE-PPO",
  "algorithm_type": "ANE_PPO",
  "parameters": {
    "learning_rate": 0.0001,
    "gamma": 0.995,
    "risk_factor": 0.5
  },
  "description": "Conservative trading strategy with lower risk",
  "is_active": true
}
```

### System Status

#### GET /api/status
Get overall system status and performance metrics.

**Response:**
```json
{
  "system_status": "online",
  "active_sessions": 1,
  "total_sessions": 10,
  "total_trades": 500,
  "recent_trades": [...],
  "performance_metrics": {
    "current_episode": 150,
    "total_episodes": 1000,
    "total_profit": 1250.50,
    "win_rate": 0.65,
    "sharpe_ratio": 1.8,
    "max_drawdown": 0.12
  },
  "timestamp": "2025-07-14T03:00:00.000000+00:00"
}
```

### Live Trading

#### POST /api/place_trade
Place a manual trade.

**Request Body:**
```json
{
  "session_id": 1,
  "symbol": "NQ",
  "action": "BUY",
  "quantity": 2,
  "price": 15000.50,
  "notes": "Manual entry based on market conditions"
}
```

### Data Management

#### POST /api/load_data
Load market data from file or download from source.

**Request Body:**
```json
{
  "symbol": "NQ",
  "start_date": "2024-01-01",
  "end_date": "2024-12-31"
}
```

### System Control

#### POST /api/shutdown
Gracefully shutdown the system, stopping all active sessions.

#### POST /api/clear_all_sessions
Clear all training sessions (for debugging purposes).

## WebSocket Events

The API also supports WebSocket connections for real-time updates at `/socket.io/`.

### Events Emitted by Server:

- `performance_metrics`: System performance metrics (CPU, Memory, GPU usage)
- `training_update`: Training progress updates
- `new_trade`: New trade notifications
- `session_update`: Session status changes
- `session_reset`: Session reset notifications

### Example WebSocket Connection:

```javascript
const socket = io('http://localhost:5000');

socket.on('connect', () => {
  console.log('Connected to server');
});

socket.on('performance_metrics', (data) => {
  console.log('Performance:', data);
});

socket.on('new_trade', (trade) => {
  console.log('New trade:', trade);
});
```

## Error Responses

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Error response format:
```json
{
  "error": "Error message description"
}
```

## Rate Limiting

Currently no rate limiting is implemented. In production, implement appropriate rate limiting to prevent abuse.

## CORS

CORS is configured to allow all origins in development. Update for production use.