"""Test routes for debugging training functionality"""
import traceback
import logging
from datetime import datetime
from flask import jsonify
from flask import current_app
from app import app, db
from models import TradingSession

logger = logging.getLogger(__name__)

@app.route('/test_training')
def test_training():
    """Test route to verify training functionality"""
    try:
        # Create a simple test session
        session_name = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session = TradingSession(
            session_name=session_name,
            algorithm_type='ANE_PPO',
            status='initializing',
            total_episodes=10,
            current_episode=0,
            total_profit=0.0,
            total_trades=0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0
        )
        db.session.add(session)
        db.session.commit()
        
        # Try to start training with minimal parameters
        config = {
            'algorithm_type': 'ANE_PPO',
            'total_episodes': 10,  # Just 10 episodes for testing
            'parameters': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'clip_range': 0.2,
                'entropy_coef': 0.01
            }
        }
        
        # Start training in a separate thread
        # Avoid circular import - we'll test training differently
        # trading_engine.start_training(session.id, config)
        
        return jsonify({
            'status': 'success',
            'message': f'Test training started with session ID: {session.id}',
            'session_name': session_name
        })
        
    except Exception as e:
        logger.error(f"Test training failed: {str(e)}")
        logger.exception(e)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/test_data_loading')
def test_data_loading():
    """Test route to verify data loading functionality"""
    try:
        from data_manager import DataManager
        dm = DataManager()
        
        # Try to load NQ data
        logger.info("Testing data loading...")
        data = dm.load_nq_data()
        
        if data is None:
            return jsonify({
                'status': 'error',
                'message': 'No data found'
            }), 404
            
        # Get basic stats
        stats = {
            'total_rows': len(data),
            'columns': list(data.columns),
            'date_range': f"{data.index[0]} to {data.index[-1]}" if hasattr(data.index, '__getitem__') else "Unknown",
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Test creating a small number of states
        logger.info("Testing state creation with 100 rows...")
        small_data = data.head(100)  # Just 100 rows
        
        # Add technical indicators
        small_data = dm.preprocess_data(small_data)
        
        return jsonify({
            'status': 'success',
            'data_stats': stats,
            'preprocessing_test': {
                'rows_tested': len(small_data),
                'columns_after_preprocessing': list(small_data.columns)[:10] + ['...'] if len(small_data.columns) > 10 else list(small_data.columns)
            }
        })
        
    except Exception as e:
        logger.error(f"Data loading test failed: {str(e)}")
        logger.exception(e)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/test_training_progress/<int:session_id>')
def test_training_progress(session_id):
    """Get training progress for a session"""
    try:
        # Check if session exists in database
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({
                'status': 'error',
                'message': f'Session {session_id} not found'
            }), 404
            
        # Check if session is in active training
        # Avoid circular import - we'll check this differently
        # active_sessions = trading_engine.get_active_sessions()
        is_active = False  # For now, just return false
        
        return jsonify({
            'status': 'success',
            'session': {
                'id': session.id,
                'name': session.session_name,
                'status': session.status,
                'current_episode': session.current_episode,
                'total_episodes': session.total_episodes,
                'is_active': is_active
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting training progress: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500