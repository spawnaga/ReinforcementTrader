import os
import logging
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SESSION_SECRET', 'dev-secret-key-change-in-production')
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///trading_system.db')
    if 'sqlite' in SQLALCHEMY_DATABASE_URI:
        # SQLite specific settings to handle concurrent access
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_pre_ping': True,
            'pool_size': 1,  # SQLite only allows one writer at a time
            'max_overflow': 0,  # Don't create extra connections
            'connect_args': {
                'check_same_thread': False,
                'timeout': 30,  # Increase timeout to 30 seconds
                'isolation_level': None  # Use autocommit mode
            }
        }
    else:
        # PostgreSQL/MySQL settings
        SQLALCHEMY_ENGINE_OPTIONS = {
            'pool_recycle': 300,
            'pool_pre_ping': True,
            'pool_size': 20,
            'max_overflow': 0
        }
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session Configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # WebSocket Configuration
    SOCKETIO_ASYNC_MODE = None  # Let Flask-SocketIO auto-detect the best mode
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ORIGINS', '*')
    SOCKETIO_LOGGER = False
    SOCKETIO_ENGINEIO_LOGGER = False
    SOCKETIO_PING_TIMEOUT = 600  # 10 minutes
    SOCKETIO_PING_INTERVAL = 300  # 5 minutes
    
    # Trading Configuration
    DEFAULT_SYMBOL = 'NQ'
    DEFAULT_TIMEFRAME = '1min'
    MAX_POSITION_SIZE = 10
    DEFAULT_RISK_LEVEL = 'moderate'
    
    # Interactive Brokers Configuration
    IB_HOST = os.environ.get('IB_HOST', '127.0.0.1')
    IB_PORT = int(os.environ.get('IB_PORT', '7497'))
    IB_CLIENT_ID = int(os.environ.get('IB_CLIENT_ID', '1'))
    IB_TIMEOUT = int(os.environ.get('IB_TIMEOUT', '10'))
    
    # Market Data Configuration
    MARKET_DATA_BUFFER_SIZE = 10000
    REAL_TIME_UPDATE_INTERVAL = 1.0  # seconds
    MARKET_DATA_RETENTION_DAYS = 30
    
    # Machine Learning Configuration
    ML_MODEL_DIR = os.environ.get('ML_MODEL_DIR', 'models')
    ML_CHECKPOINT_INTERVAL = 100  # episodes
    ML_MAX_EPISODES = 10000
    ML_LEARNING_RATE = float(os.environ.get('ML_LEARNING_RATE', '3e-4'))
    ML_GAMMA = float(os.environ.get('ML_GAMMA', '0.99'))
    ML_CLIP_RANGE = float(os.environ.get('ML_CLIP_RANGE', '0.2'))
    ML_ENTROPY_COEF = float(os.environ.get('ML_ENTROPY_COEF', '0.01'))
    ML_VALUE_LOSS_COEF = float(os.environ.get('ML_VALUE_LOSS_COEF', '0.5'))
    ML_MAX_GRAD_NORM = float(os.environ.get('ML_MAX_GRAD_NORM', '0.5'))
    ML_N_STEPS = int(os.environ.get('ML_N_STEPS', '2048'))
    ML_BATCH_SIZE = int(os.environ.get('ML_BATCH_SIZE', '64'))
    ML_N_EPOCHS = int(os.environ.get('ML_N_EPOCHS', '10'))
    
    # GPU Configuration
    USE_GPU = os.environ.get('USE_GPU', 'True').lower() == 'true'
    GPU_MEMORY_FRACTION = float(os.environ.get('GPU_MEMORY_FRACTION', '0.8'))
    
    # Genetic Algorithm Configuration
    GA_POPULATION_SIZE = int(os.environ.get('GA_POPULATION_SIZE', '50'))
    GA_MUTATION_RATE = float(os.environ.get('GA_MUTATION_RATE', '0.1'))
    GA_CROSSOVER_RATE = float(os.environ.get('GA_CROSSOVER_RATE', '0.8'))
    GA_ELITE_RATIO = float(os.environ.get('GA_ELITE_RATIO', '0.1'))
    GA_MAX_GENERATIONS = int(os.environ.get('GA_MAX_GENERATIONS', '100'))
    
    # Risk Management Configuration
    MAX_DAILY_LOSS = float(os.environ.get('MAX_DAILY_LOSS', '1000.0'))
    MAX_DRAWDOWN = float(os.environ.get('MAX_DRAWDOWN', '0.15'))
    MAX_LEVERAGE = float(os.environ.get('MAX_LEVERAGE', '10.0'))
    MAX_CORRELATION = float(os.environ.get('MAX_CORRELATION', '0.7'))
    VAR_LIMIT = float(os.environ.get('VAR_LIMIT', '0.02'))
    MAX_CONCENTRATION = float(os.environ.get('MAX_CONCENTRATION', '0.25'))
    
    # NQ Futures Specifications
    NQ_TICK_SIZE = 0.25
    NQ_VALUE_PER_TICK = 5.0
    NQ_CONTRACT_SIZE = 20
    NQ_MARGIN_REQUIREMENT = float(os.environ.get('NQ_MARGIN_REQUIREMENT', '15000.0'))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    LOG_FILE = os.environ.get('LOG_FILE', 'trading_system.log')
    LOG_MAX_BYTES = int(os.environ.get('LOG_MAX_BYTES', '10485760'))  # 10MB
    LOG_BACKUP_COUNT = int(os.environ.get('LOG_BACKUP_COUNT', '5'))
    
    # Performance Monitoring
    ENABLE_PROFILING = os.environ.get('ENABLE_PROFILING', 'False').lower() == 'true'
    PERFORMANCE_METRICS_INTERVAL = int(os.environ.get('PERFORMANCE_METRICS_INTERVAL', '5'))
    
    # Security Configuration
    WTF_CSRF_ENABLED = True
    WTF_CSRF_TIME_LIMIT = 3600
    
    # API Rate Limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    RATELIMIT_DEFAULT = "100 per hour"
    
    # Backup and Recovery
    BACKUP_ENABLED = os.environ.get('BACKUP_ENABLED', 'True').lower() == 'true'
    BACKUP_INTERVAL_HOURS = int(os.environ.get('BACKUP_INTERVAL_HOURS', '24'))
    BACKUP_RETENTION_DAYS = int(os.environ.get('BACKUP_RETENTION_DAYS', '30'))
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, Config.LOG_LEVEL),
            format=Config.LOG_FORMAT,
            datefmt=Config.LOG_DATE_FORMAT
        )
        
        # Create necessary directories
        os.makedirs(Config.ML_MODEL_DIR, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('backups', exist_ok=True)
        
        # Setup file logging
        if Config.LOG_FILE:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                f'logs/{Config.LOG_FILE}',
                maxBytes=Config.LOG_MAX_BYTES,
                backupCount=Config.LOG_BACKUP_COUNT
            )
            file_handler.setFormatter(logging.Formatter(
                Config.LOG_FORMAT,
                Config.LOG_DATE_FORMAT
            ))
            app.logger.addHandler(file_handler)
        
        # Validate critical environment variables
        Config._validate_config()
    
    @staticmethod
    def _validate_config():
        """Validate critical configuration values"""
        errors = []
        
        # Validate numeric ranges
        if not 0 < Config.ML_LEARNING_RATE < 1:
            errors.append("ML_LEARNING_RATE must be between 0 and 1")
        
        if not 0 < Config.ML_GAMMA <= 1:
            errors.append("ML_GAMMA must be between 0 and 1")
        
        if not 0 < Config.ML_CLIP_RANGE < 1:
            errors.append("ML_CLIP_RANGE must be between 0 and 1")
        
        if not 0 <= Config.MAX_DRAWDOWN <= 1:
            errors.append("MAX_DRAWDOWN must be between 0 and 1")
        
        if not 0 < Config.GA_MUTATION_RATE < 1:
            errors.append("GA_MUTATION_RATE must be between 0 and 1")
        
        if not 0 < Config.GA_CROSSOVER_RATE < 1:
            errors.append("GA_CROSSOVER_RATE must be between 0 and 1")
        
        # Validate required directories can be created
        try:
            os.makedirs(Config.ML_MODEL_DIR, exist_ok=True)
            os.makedirs('logs', exist_ok=True)
        except PermissionError as e:
            errors.append(f"Cannot create required directories: {e}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'
    
    # Disable CSRF for development
    WTF_CSRF_ENABLED = False
    
    # Allow all origins for CORS in development
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    
    # Enable profiling in development
    ENABLE_PROFILING = True
    
    # Shorter session timeout for development
    PERMANENT_SESSION_LIFETIME = timedelta(hours=1)

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    # Strict security in production
    SESSION_COOKIE_SECURE = True
    WTF_CSRF_ENABLED = True
    
    # Specific CORS origins in production
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',')
    
    # Disable profiling in production
    ENABLE_PROFILING = False
    
    # Longer session timeout for production
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Enhanced security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' cdn.jsdelivr.net cdnjs.cloudflare.com d3js.org; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; font-src 'self' cdnjs.cloudflare.com; img-src 'self' data:; connect-src 'self' ws: wss:;"
    }

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    
    # Use in-memory database for testing
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Disable CSRF for testing
    WTF_CSRF_ENABLED = False
    
    # Shorter timeouts for testing
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=5)
    IB_TIMEOUT = 1
    
    # Reduced batch sizes for faster testing
    ML_N_STEPS = 64
    ML_BATCH_SIZE = 8
    GA_POPULATION_SIZE = 10
    GA_MAX_GENERATIONS = 5
    
    # Disable backup in testing
    BACKUP_ENABLED = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])

# Trading Strategy Configurations
STRATEGY_CONFIGS = {
    'conservative': {
        'max_position_size': 2,
        'max_daily_loss': 500.0,
        'max_drawdown': 0.05,
        'risk_per_trade': 0.01,
        'stop_loss_multiplier': 0.5,
        'take_profit_multiplier': 1.5
    },
    'moderate': {
        'max_position_size': 5,
        'max_daily_loss': 1000.0,
        'max_drawdown': 0.10,
        'risk_per_trade': 0.02,
        'stop_loss_multiplier': 1.0,
        'take_profit_multiplier': 2.0
    },
    'aggressive': {
        'max_position_size': 10,
        'max_daily_loss': 2000.0,
        'max_drawdown': 0.20,
        'risk_per_trade': 0.05,
        'stop_loss_multiplier': 2.0,
        'take_profit_multiplier': 3.0
    }
}

# Algorithm Default Parameters
ALGORITHM_DEFAULTS = {
    'ANE_PPO': {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'genetic_population_size': 50,
        'genetic_mutation_rate': 0.1,
        'genetic_crossover_rate': 0.8,
        'attention_heads': 8,
        'attention_dim': 256,
        'transformer_layers': 6
    },
    'PPO': {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 0.5,
        'max_grad_norm': 0.5,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10
    },
    'DQN': {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 100000,
        'batch_size': 32,
        'target_update': 1000,
        'multi_step': 3
    },
    'GENETIC': {
        'population_size': 50,
        'mutation_rate': 0.1,
        'crossover_rate': 0.8,
        'elite_ratio': 0.1,
        'max_generations': 100,
        'convergence_threshold': 1e-6
    }
}

# Market Hours Configuration (EST)
MARKET_HOURS = {
    'NQ': {
        'sunday': {'start': '18:00', 'end': '23:59'},
        'monday': {'start': '00:00', 'end': '17:00'},
        'tuesday': {'start': '18:00', 'end': '23:59'},
        'wednesday': {'start': '00:00', 'end': '17:00'},
        'thursday': {'start': '18:00', 'end': '23:59'},
        'friday': {'start': '00:00', 'end': '17:00'},
        'saturday': {'start': '18:00', 'end': '23:59'},
        'maintenance': {
            'start': '17:00',
            'end': '18:00',
            'days': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
        }
    }
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'technical_indicators': [
        'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100',
        'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_100',
        'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'stoch_k', 'stoch_d', 'atr_14', 'adx_14'
    ],
    'price_features': [
        'price_change', 'price_change_pct', 'price_range',
        'high_low_ratio', 'open_close_ratio', 'volume_ratio',
        'volatility_5', 'volatility_20', 'volatility_ratio'
    ],
    'time_features': [
        'hour', 'minute', 'day_of_week', 'is_session_start',
        'is_session_end', 'time_since_open', 'time_to_close'
    ],
    'market_microstructure': [
        'tick_direction', 'uptick_ratio', 'volume_price_trend',
        'price_acceleration', 'momentum_5', 'momentum_20'
    ],
    'regime_features': [
        'trend_strength', 'market_regime', 'volatility_regime',
        'volume_regime', 'correlation_regime'
    ]
}
