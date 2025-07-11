from extensions import db
from datetime import datetime
from sqlalchemy import Text, Float, Integer, DateTime, String, Boolean, JSON

class TradingSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_name = db.Column(db.String(100), nullable=False)
    algorithm_type = db.Column(db.String(50), nullable=False)
    parameters = db.Column(JSON)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    status = db.Column(db.String(20), default='active')
    total_episodes = db.Column(db.Integer, default=0)
    current_episode = db.Column(db.Integer, default=0)
    total_profit = db.Column(db.Float, default=0.0)
    total_trades = db.Column(db.Integer, default=0)
    win_rate = db.Column(db.Float, default=0.0)
    sharpe_ratio = db.Column(db.Float, default=0.0)
    max_drawdown = db.Column(db.Float, default=0.0)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('trading_session.id'), nullable=False)
    trade_id = db.Column(db.String(50), nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime)
    entry_price = db.Column(db.Float, nullable=False)
    exit_price = db.Column(db.Float)
    position_type = db.Column(db.String(10), nullable=False)  # 'long' or 'short'
    quantity = db.Column(db.Integer, default=1)
    profit_loss = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='open')
    episode_number = db.Column(db.Integer, nullable=False)

class MarketData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    symbol = db.Column(db.String(10), nullable=False)
    open_price = db.Column(db.Float, nullable=False)
    high_price = db.Column(db.Float, nullable=False)
    low_price = db.Column(db.Float, nullable=False)
    close_price = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Integer, nullable=False)
    timeframe = db.Column(db.String(10), default='1min')

class TrainingMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('trading_session.id'), nullable=False)
    episode = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    reward = db.Column(db.Float, nullable=False)
    loss = db.Column(db.Float)
    epsilon = db.Column(db.Float)
    learning_rate = db.Column(db.Float)
    action_distribution = db.Column(JSON)
    network_weights_summary = db.Column(JSON)

class AlgorithmConfig(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    algorithm_type = db.Column(db.String(50), nullable=False)
    parameters = db.Column(JSON, nullable=False)
    description = db.Column(Text)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
