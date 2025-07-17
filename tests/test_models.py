"""
Tests for database models
"""

import pytest
from datetime import datetime
from models import MarketData, TradingSession, Trade, TrainingMetrics, AlgorithmConfig
from app import app, db

class TestDatabaseModels:
    
    @pytest.fixture
    def app_context(self):
        """Create application context for database operations"""
        with app.app_context():
            db.create_all()
            yield
            db.session.rollback()
            db.drop_all()
    
    def test_market_data_model(self, app_context):
        """Test MarketData model"""
        market_data = MarketData(
            ticker='NQ',
            timestamp=datetime.now(),
            open_price=4000.0,
            high_price=4050.0,
            low_price=3950.0,
            close_price=4025.0,
            volume=10000
        )
        
        db.session.add(market_data)
        db.session.commit()
        
        # Query back
        queried = MarketData.query.first()
        assert queried is not None
        assert queried.ticker == 'NQ'
        assert queried.close_price == 4025.0
    
    def test_trading_session_model(self, app_context):
        """Test TradingSession model"""
        session = TradingSession(
            session_name='Test Session',
            algorithm_type='ANE_PPO',
            status='active',
            total_episodes=1000,
            current_episode=100
        )
        
        db.session.add(session)
        db.session.commit()
        
        # Query back
        queried = TradingSession.query.first()
        assert queried is not None
        assert queried.session_name == 'Test Session'
        assert queried.algorithm_type == 'ANE_PPO'
        assert queried.total_profit == 0.0  # Default value
    
    def test_trade_model(self, app_context):
        """Test Trade model with session relationship"""
        # Create session first
        session = TradingSession(
            session_name='Test Session',
            algorithm_type='DQN',
            status='active'
        )
        db.session.add(session)
        db.session.commit()
        
        # Create trade
        trade = Trade(
            session_id=session.id,
            position_type='long',
            entry_price=4000.0,
            exit_price=4050.0,
            position_size=2,
            profit_loss=100.0,
            entry_time=datetime.now()
        )
        
        db.session.add(trade)
        db.session.commit()
        
        # Query back
        queried = Trade.query.first()
        assert queried is not None
        assert queried.position_type == 'long'
        assert queried.profit_loss == 100.0
        assert queried.session_id == session.id
        
        # Test relationship
        assert queried.session.session_name == 'Test Session'
    
    def test_training_metrics_model(self, app_context):
        """Test TrainingMetrics model"""
        # Create session first
        session = TradingSession(
            session_name='Test Session',
            algorithm_type='ANE_PPO',
            status='active'
        )
        db.session.add(session)
        db.session.commit()
        
        # Create metrics
        metrics = TrainingMetrics(
            session_id=session.id,
            episode=50,
            reward=150.5,
            loss=0.05,
            learning_rate=0.001
        )
        
        db.session.add(metrics)
        db.session.commit()
        
        # Query back
        queried = TrainingMetrics.query.first()
        assert queried is not None
        assert queried.episode == 50
        assert queried.reward == 150.5
    
    def test_algorithm_config_model(self, app_context):
        """Test AlgorithmConfig model"""
        config = AlgorithmConfig(
            algorithm_type='ANE_PPO',
            parameters={
                'learning_rate': 0.0003,
                'batch_size': 64,
                'transformer_layers': 6
            },
            description='Test configuration'
        )
        
        db.session.add(config)
        db.session.commit()
        
        # Query back
        queried = AlgorithmConfig.query.first()
        assert queried is not None
        assert queried.algorithm_type == 'ANE_PPO'
        assert queried.parameters['learning_rate'] == 0.0003
    
    def test_cascade_delete(self, app_context):
        """Test cascade deletion of related records"""
        # Create session with trades
        session = TradingSession(
            session_name='Test Session',
            algorithm_type='DQN',
            status='completed'
        )
        db.session.add(session)
        db.session.commit()
        
        # Add multiple trades
        for i in range(3):
            trade = Trade(
                session_id=session.id,
                position_type='long' if i % 2 == 0 else 'short',
                entry_price=4000.0 + i * 10,
                exit_price=4050.0 + i * 10,
                position_size=1,
                profit_loss=50.0
            )
            db.session.add(trade)
        
        db.session.commit()
        
        # Verify trades exist
        assert Trade.query.count() == 3
        
        # Delete session
        db.session.delete(session)
        db.session.commit()
        
        # Trades should be deleted too
        assert Trade.query.count() == 0
    
    def test_session_statistics_update(self, app_context):
        """Test updating session statistics"""
        session = TradingSession(
            session_name='Test Session',
            algorithm_type='ANE_PPO',
            status='active'
        )
        db.session.add(session)
        db.session.commit()
        
        # Update statistics
        session.total_trades = 10
        session.winning_trades = 6
        session.total_profit = 500.0
        session.max_drawdown = -200.0
        
        db.session.commit()
        
        # Query back
        queried = TradingSession.query.first()
        assert queried.total_trades == 10
        assert queried.winning_trades == 6
        assert queried.win_rate == 60.0  # Should be calculated