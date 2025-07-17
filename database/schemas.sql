-- Professional PostgreSQL Schema for RL Trading System
-- Designed for comprehensive tracking and performance analysis

-- 1. Training Sessions Table
CREATE TABLE IF NOT EXISTS training_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,
    algorithm_name VARCHAR(100) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    total_episodes INTEGER DEFAULT 0,
    total_steps BIGINT DEFAULT 0,
    final_reward DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'running', -- running, completed, failed
    hyperparameters JSONB,
    gpu_info JSONB,
    notes TEXT
);

-- 2. Episode Metrics Table (Aggregated per episode)
CREATE TABLE IF NOT EXISTS episode_metrics (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    total_reward DECIMAL(10, 2),
    total_profit DECIMAL(10, 2),
    num_trades INTEGER DEFAULT 0,
    num_winning_trades INTEGER DEFAULT 0,
    num_losing_trades INTEGER DEFAULT 0,
    max_drawdown DECIMAL(10, 2),
    sharpe_ratio DECIMAL(8, 4),
    win_rate DECIMAL(5, 4),
    avg_trade_duration_minutes INTEGER,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    steps INTEGER,
    UNIQUE(session_id, episode_number)
);

-- 3. Trades Table (Detailed trade tracking)
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    episode_number INTEGER,
    trade_number INTEGER,
    entry_time TIMESTAMP WITH TIME ZONE,
    exit_time TIMESTAMP WITH TIME ZONE,
    position_type VARCHAR(10), -- LONG, SHORT
    entry_price DECIMAL(10, 2),
    exit_price DECIMAL(10, 2),
    quantity INTEGER DEFAULT 1,
    gross_profit DECIMAL(10, 2),
    commission DECIMAL(8, 2),
    slippage DECIMAL(8, 2),
    net_profit DECIMAL(10, 2),
    profit_ticks DECIMAL(10, 2),
    trade_duration_minutes INTEGER,
    max_profit DECIMAL(10, 2),
    max_loss DECIMAL(10, 2),
    exit_reason VARCHAR(50), -- stop_loss, take_profit, signal, episode_end
    state_features JSONB -- Store state at entry for analysis
);

-- 4. Position Snapshots (Track positions over time)
CREATE TABLE IF NOT EXISTS position_snapshots (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE,
    episode_number INTEGER,
    position INTEGER, -- -1, 0, 1
    current_price DECIMAL(10, 2),
    unrealized_pnl DECIMAL(10, 2),
    account_value DECIMAL(12, 2),
    margin_used DECIMAL(10, 2),
    exposure DECIMAL(10, 2)
);

-- 5. Algorithm Performance (Track algorithm decisions)
CREATE TABLE IF NOT EXISTS algorithm_metrics (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    episode_number INTEGER,
    step_number INTEGER,
    action_taken VARCHAR(10), -- BUY, SELL, HOLD
    action_probabilities JSONB, -- {buy: 0.3, sell: 0.2, hold: 0.5}
    q_values JSONB, -- For DQN algorithms
    policy_loss DECIMAL(10, 6),
    value_loss DECIMAL(10, 6),
    entropy DECIMAL(10, 6),
    learning_rate DECIMAL(10, 8),
    epsilon DECIMAL(5, 4), -- For exploration
    reward DECIMAL(10, 2),
    advantage DECIMAL(10, 2) -- For PPO
);

-- 6. Model Checkpoints (Track model versions)
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    checkpoint_name VARCHAR(200) UNIQUE NOT NULL,
    episode_number INTEGER,
    total_steps BIGINT,
    avg_reward DECIMAL(10, 2),
    best_reward DECIMAL(10, 2),
    validation_score DECIMAL(10, 4),
    model_path VARCHAR(500),
    model_size_mb DECIMAL(8, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    is_best BOOLEAN DEFAULT FALSE,
    notes TEXT
);

-- 7. Learning Progress (Track learning curve)
CREATE TABLE IF NOT EXISTS learning_progress (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) REFERENCES training_sessions(session_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    episode_number INTEGER,
    rolling_avg_reward DECIMAL(10, 2), -- 100-episode rolling average
    rolling_avg_profit DECIMAL(10, 2),
    rolling_sharpe_ratio DECIMAL(8, 4),
    rolling_win_rate DECIMAL(5, 4),
    exploration_rate DECIMAL(5, 4),
    gradient_norm DECIMAL(10, 6),
    value_estimate_accuracy DECIMAL(5, 4)
);

-- Create indexes for fast queries
CREATE INDEX idx_episode_metrics_session ON episode_metrics(session_id);
CREATE INDEX idx_trades_session_episode ON trades(session_id, episode_number);
CREATE INDEX idx_trades_profit ON trades(net_profit);
CREATE INDEX idx_algorithm_metrics_session ON algorithm_metrics(session_id, episode_number);
CREATE INDEX idx_learning_progress_session ON learning_progress(session_id, episode_number);
CREATE INDEX idx_position_snapshots_session ON position_snapshots(session_id, timestamp);

-- Create views for quick analysis
CREATE OR REPLACE VIEW session_summary AS
SELECT 
    ts.session_id,
    ts.algorithm_name,
    ts.ticker,
    ts.start_time,
    ts.total_episodes,
    COUNT(DISTINCT t.id) as total_trades,
    SUM(t.net_profit) as total_profit,
    AVG(em.sharpe_ratio) as avg_sharpe_ratio,
    AVG(em.win_rate) as avg_win_rate,
    MAX(em.total_reward) as best_episode_reward,
    ts.status
FROM training_sessions ts
LEFT JOIN trades t ON ts.session_id = t.session_id
LEFT JOIN episode_metrics em ON ts.session_id = em.session_id
GROUP BY ts.session_id, ts.algorithm_name, ts.ticker, ts.start_time, ts.total_episodes, ts.status;

-- View for recent performance
CREATE OR REPLACE VIEW recent_performance AS
SELECT 
    lp.session_id,
    lp.episode_number,
    lp.rolling_avg_reward,
    lp.rolling_avg_profit,
    lp.rolling_sharpe_ratio,
    lp.rolling_win_rate,
    ts.algorithm_name
FROM learning_progress lp
JOIN training_sessions ts ON lp.session_id = ts.session_id
WHERE lp.episode_number > (SELECT MAX(episode_number) - 10 FROM learning_progress WHERE session_id = lp.session_id)
ORDER BY lp.session_id, lp.episode_number DESC;