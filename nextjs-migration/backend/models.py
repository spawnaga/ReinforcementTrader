"""
Pydantic models for FastAPI
Type-safe data validation and serialization
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

# Enums
class AlgorithmType(str, Enum):
    ANE_PPO = "ane-ppo"
    DQN = "dqn"
    GENETIC = "genetic"

class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

class PositionType(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

# Request/Response Models
class TrainingStartRequest(BaseModel):
    algorithm_type: AlgorithmType = AlgorithmType.ANE_PPO
    total_episodes: int = Field(default=1000, ge=10, le=10000)
    symbol: str = "NQ"
    transformer_layers: Optional[int] = Field(default=2, ge=1, le=8)
    attention_dim: Optional[int] = Field(default=256, ge=64, le=512)
    learning_rate: Optional[float] = Field(default=3e-4, gt=0, le=1)
    
    @validator('total_episodes')
    def validate_episodes(cls, v):
        # Ensure episodes are in steps of 10
        return (v // 10) * 10

class TrainingSession(BaseModel):
    id: str
    algorithm_type: AlgorithmType
    status: SessionStatus
    total_episodes: int
    current_episode: int
    created_at: datetime
    updated_at: datetime
    symbol: str
    config: Optional[Dict[str, Any]] = {}

class Trade(BaseModel):
    id: int
    session_id: str
    timestamp: datetime
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_type: PositionType
    quantity: int = 1
    profit: Optional[float]
    commission: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_time is None

class MarketData(BaseModel):
    timestamp: datetime
    open: float = Field(alias="open_price")
    high: float = Field(alias="high_price")
    low: float = Field(alias="low_price")
    close: float = Field(alias="close_price")
    volume: int
    
    class Config:
        allow_population_by_field_name = True

class PerformanceMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    network_io: float
    timestamp: datetime = Field(default_factory=datetime.now)

class TrainingMetrics(BaseModel):
    episode: int
    reward: float
    loss: float
    win_rate: Optional[float]
    profit: Optional[float]
    timestamp: datetime = Field(default_factory=datetime.now)

class SessionSummary(BaseModel):
    session_id: str
    total_trades: int
    profitable_trades: int
    total_profit: float
    win_rate: float
    avg_profit_per_trade: float
    max_drawdown: float
    sharpe_ratio: Optional[float]

# WebSocket Messages
class WSMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class WSTrainingUpdate(WSMessage):
    type: str = "training_update"
    data: TrainingMetrics

class WSTradeUpdate(WSMessage):
    type: str = "trade_update"
    data: Trade

class WSPerformanceUpdate(WSMessage):
    type: str = "performance_update"
    data: PerformanceMetrics

# API Responses
class APIResponse(BaseModel):
    success: bool
    message: Optional[str]
    data: Optional[Any]

class SessionListResponse(BaseModel):
    sessions: List[TrainingSession]
    count: int

class TradeListResponse(BaseModel):
    trades: List[Trade]
    count: int
    summary: Optional[SessionSummary]