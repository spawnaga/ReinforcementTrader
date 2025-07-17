"""
Tests for API routes
"""

import pytest
import json
from app import app, db
from models import TradingSession, Trade, MarketData
from datetime import datetime

class TestAPIRoutes:
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.test_client() as client:
            with app.app_context():
                db.create_all()
            yield client
            with app.app_context():
                db.drop_all()
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'database' in data
        assert 'timestamp' in data
    
    def test_get_sessions_empty(self, client):
        """Test getting sessions when none exist"""
        response = client.get('/api/sessions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert isinstance(data, list)
        assert len(data) == 0
    
    def test_get_sessions_with_data(self, client):
        """Test getting sessions with data"""
        with app.app_context():
            # Create test sessions
            session1 = TradingSession(
                session_name='Test Session 1',
                algorithm_type='ANE_PPO',
                status='active'
            )
            session2 = TradingSession(
                session_name='Test Session 2',
                algorithm_type='DQN',
                status='completed'
            )
            db.session.add_all([session1, session2])
            db.session.commit()
        
        response = client.get('/api/sessions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2
        assert data[0]['session_name'] == 'Test Session 1'
        assert data[1]['status'] == 'completed'
    
    def test_get_session_by_id(self, client):
        """Test getting specific session"""
        with app.app_context():
            session = TradingSession(
                session_name='Test Session',
                algorithm_type='ANE_PPO',
                status='active'
            )
            db.session.add(session)
            db.session.commit()
            session_id = session.id
        
        response = client.get(f'/api/sessions/{session_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['session_name'] == 'Test Session'
    
    def test_get_nonexistent_session(self, client):
        """Test getting non-existent session"""
        response = client.get('/api/sessions/999')
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_create_session(self, client):
        """Test creating a new session"""
        session_data = {
            'session_name': 'New Session',
            'algorithm_type': 'ANE_PPO',
            'total_episodes': 1000,
            'parameters': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        }
        
        response = client.post('/api/sessions',
                             json=session_data,
                             content_type='application/json')
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['session_name'] == 'New Session'
        assert 'id' in data
    
    def test_update_session(self, client):
        """Test updating a session"""
        with app.app_context():
            session = TradingSession(
                session_name='Original Name',
                algorithm_type='DQN',
                status='active'
            )
            db.session.add(session)
            db.session.commit()
            session_id = session.id
        
        update_data = {
            'session_name': 'Updated Name',
            'status': 'completed',
            'total_profit': 1000.0
        }
        
        response = client.put(f'/api/sessions/{session_id}',
                            json=update_data,
                            content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['session_name'] == 'Updated Name'
        assert data['status'] == 'completed'
    
    def test_delete_session(self, client):
        """Test deleting a session"""
        with app.app_context():
            session = TradingSession(
                session_name='To Delete',
                algorithm_type='DQN',
                status='completed'
            )
            db.session.add(session)
            db.session.commit()
            session_id = session.id
        
        response = client.delete(f'/api/sessions/{session_id}')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Session deleted'
        
        # Verify deletion
        response = client.get(f'/api/sessions/{session_id}')
        assert response.status_code == 404
    
    def test_get_trades(self, client):
        """Test getting trades"""
        with app.app_context():
            # Create session and trades
            session = TradingSession(
                session_name='Test Session',
                algorithm_type='ANE_PPO',
                status='active'
            )
            db.session.add(session)
            db.session.commit()
            
            trade1 = Trade(
                session_id=session.id,
                position_type='long',
                entry_price=4000.0,
                exit_price=4050.0,
                profit_loss=50.0
            )
            trade2 = Trade(
                session_id=session.id,
                position_type='short',
                entry_price=4100.0,
                exit_price=4080.0,
                profit_loss=20.0
            )
            db.session.add_all([trade1, trade2])
            db.session.commit()
        
        response = client.get('/api/trades')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2
        assert data[0]['profit_loss'] == 50.0
    
    def test_get_recent_trades(self, client):
        """Test getting recent trades with limit"""
        with app.app_context():
            session = TradingSession(
                session_name='Test Session',
                algorithm_type='ANE_PPO',
                status='active'
            )
            db.session.add(session)
            db.session.commit()
            
            # Create 5 trades
            for i in range(5):
                trade = Trade(
                    session_id=session.id,
                    position_type='long',
                    entry_price=4000.0 + i,
                    profit_loss=10.0 * i
                )
                db.session.add(trade)
            db.session.commit()
        
        response = client.get('/api/recent_trades?limit=3')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 3  # Should respect limit
    
    def test_get_market_data(self, client):
        """Test getting market data"""
        with app.app_context():
            # Create market data
            for i in range(3):
                market_data = MarketData(
                    ticker='NQ',
                    timestamp=datetime.now(),
                    open_price=4000.0 + i,
                    high_price=4050.0 + i,
                    low_price=3950.0 + i,
                    close_price=4025.0 + i,
                    volume=1000 + i
                )
                db.session.add(market_data)
            db.session.commit()
        
        response = client.get('/api/market_data?ticker=NQ')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 3
        assert all(d['ticker'] == 'NQ' for d in data)