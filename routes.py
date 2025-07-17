from flask import request, jsonify
from app import app, trading_engine
from app import db, socketio
from models import TradingSession, Trade, MarketData, TrainingMetrics, AlgorithmConfig
from datetime import datetime, timezone
import json
import logging
from db_utils import retry_on_db_error
import pandas as pd
from data_manager import DataManager

# Import websocket handlers to register them
import websocket_handler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API Health Check and Root
# ---------------------------------------------------------------------------

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - returns API information"""
    return jsonify({
        'name': 'Revolutionary AI Trading System API',
        'version': '2.0.0',
        'type': 'Backend API',
        'documentation': '/api/docs',
        'health': '/health',
        'endpoints': {
            'sessions': '/api/sessions',
            'trades': '/api/trades',
            'training': '/api/start_training',
            'status': '/api/status'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })

@app.route('/training_dashboard')
def training_dashboard():
    """Serve the advanced training dashboard"""
    return app.send_static_file('training_dashboard.html')

# ---------------------------------------------------------------------------
# Training Control API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/start_training', methods=['POST'])
@retry_on_db_error()
def start_training():
    """Start a new training session"""
    try:
        logger.info("Received training start request")
        data = request.get_json()
        logger.info(f"Training request data: {data}")
        
        # Create new trading session
        session = TradingSession(
            session_name=data.get('session_name', f'Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            algorithm_type=data.get('algorithm_type', 'ANE_PPO'),
            parameters=data.get('parameters', {}),
            total_episodes=data.get('total_episodes', 1000)
        )
        
        db.session.add(session)
        db.session.commit()
        logger.info(f"Created training session with ID: {session.id}")
        
        # Start training in background
        success = trading_engine.start_training(session.id, data)
        
        if success:
            logger.info(f"Training started successfully for session {session.id}")
            return jsonify({
                'success': True,
                'session_id': session.id,
                'message': 'Training started successfully'
            })
        else:
            logger.error(f"Failed to start training for session {session.id}")
            return jsonify({
                'success': False,
                'error': 'Failed to start training'
            }), 500
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stop_training', methods=['POST'])
@retry_on_db_error()
def stop_training():
    """Stop a training session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        session = TradingSession.query.get(session_id)
        if session:
            session.status = 'stopped'
            session.end_time = datetime.now(timezone.utc)
            db.session.commit()
            
            trading_engine.stop_training(session_id)
            
            return jsonify({
                'success': True,
                'message': 'Training stopped successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Session not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Error stopping training: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ---------------------------------------------------------------------------
# Session Management API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/sessions', methods=['GET', 'POST'])
def handle_sessions():
    """Get all trading sessions or create a new one"""
    if request.method == 'GET':
        try:
            sessions = TradingSession.query.order_by(TradingSession.start_time.desc()).all()
            result = []
            
            for session in sessions:
                result.append({
                    'id': session.id,
                    'name': session.session_name,
                    'algorithm_type': session.algorithm_type,
                    'status': session.status,
                    'start_time': session.start_time.isoformat() if session.start_time else None,
                    'end_time': session.end_time.isoformat() if session.end_time else None,
                    'total_episodes': session.total_episodes,
                    'current_episode': session.current_episode,
                    'total_profit': session.total_profit,
                    'total_trades': session.total_trades,
                    'win_rate': session.win_rate,
                    'sharpe_ratio': session.sharpe_ratio,
                    'max_drawdown': session.max_drawdown
                })
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Error fetching sessions: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    else:  # POST - Create new session
        try:
            data = request.get_json()
            
            # Stop any existing active sessions
            active_sessions = TradingSession.query.filter_by(status='active').all()
            for session in active_sessions:
                session.status = 'stopped'
                session.end_time = datetime.now(timezone.utc)
                trading_engine.stop_training(session.id)
            
            # Create new session with fresh start
            new_session = TradingSession(
                session_name=data.get('session_name', f'Training Session {datetime.now().strftime("%Y%m%d_%H%M%S")}'),
                algorithm_type=data.get('algorithm_type', 'ANE_PPO'),
                status='active',
                start_time=datetime.now(timezone.utc),
                total_episodes=data.get('total_episodes', 1000),
                current_episode=0,
                total_profit=0.0,
                total_trades=0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                parameters=data.get('parameters', {})
            )
            
            db.session.add(new_session)
            db.session.commit()
            
            return jsonify({
                'id': new_session.id,
                'name': new_session.session_name,
                'algorithm_type': new_session.algorithm_type,
                'status': new_session.status,
                'total_episodes': new_session.total_episodes
            })
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>', methods=['GET', 'DELETE'])
def handle_session(session_id):
    """Get or delete a specific session"""
    if request.method == 'GET':
        try:
            session = TradingSession.query.get(session_id)
            if not session:
                return jsonify({'error': 'Session not found'}), 404
                
            return jsonify({
                'id': session.id,
                'name': session.session_name,
                'algorithm_type': session.algorithm_type,
                'status': session.status,
                'start_time': session.start_time.isoformat() if session.start_time else None,
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'total_episodes': session.total_episodes,
                'current_episode': session.current_episode,
                'total_profit': session.total_profit,
                'total_trades': session.total_trades,
                'win_rate': session.win_rate,
                'sharpe_ratio': session.sharpe_ratio,
                'max_drawdown': session.max_drawdown,
                'parameters': session.parameters
            })
            
        except Exception as e:
            logger.error(f"Error fetching session: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    else:  # DELETE
        try:
            session = TradingSession.query.get(session_id)
            if not session:
                return jsonify({'error': 'Session not found'}), 404
                
            # Stop training if active
            if session.status == 'active':
                trading_engine.stop_training(session_id)
                
            # Delete related data
            Trade.query.filter_by(session_id=session_id).delete()
            TrainingMetrics.query.filter_by(session_id=session_id).delete()
            
            db.session.delete(session)
            db.session.commit()
            
            return jsonify({'success': True, 'message': 'Session deleted successfully'})
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/start', methods=['POST'])
@retry_on_db_error()
def start_session(session_id):
    """Start a training session"""
    try:
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
            
        # Get data configuration from request if provided
        data = request.get_json() or {}
        
        # Start training with the session parameters
        # Prepare training configuration
        config = {
            'algorithm_type': session.algorithm_type,
            'total_episodes': session.total_episodes,
            'parameters': session.parameters or {}
        }
        
        # Add data configuration if provided
        if 'dataConfig' in data:
            config['parameters']['dataConfig'] = data['dataConfig']
        
        success = trading_engine.start_training(session_id, config)
        
        if success:
            session.status = 'active'
            db.session.commit()
            return jsonify({'success': True, 'message': 'Training started'})
        else:
            return jsonify({'error': 'Failed to start training'}), 500
            
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/pause', methods=['POST'])
@retry_on_db_error()
def pause_session(session_id):
    """Pause a training session"""
    try:
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
            
        trading_engine.pause_training(session_id)
        session.status = 'paused'
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Training paused'})
        
    except Exception as e:
        logger.error(f"Error pausing session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/stop', methods=['POST'])
@retry_on_db_error()
def stop_session(session_id):
    """Stop a training session"""
    try:
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
            
        trading_engine.stop_training(session_id)
        session.status = 'stopped'
        session.end_time = datetime.now(timezone.utc)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Training stopped'})
        
    except Exception as e:
        logger.error(f"Error stopping session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<int:session_id>/reset', methods=['POST'])
@retry_on_db_error()
def reset_session(session_id):
    """Reset a training session - clear all trades and metrics"""
    try:
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
            
        # Delete all trades for this session
        Trade.query.filter_by(session_id=session_id).delete()
        
        # Delete all training metrics
        TrainingMetrics.query.filter_by(session_id=session_id).delete()
        
        # Reset session stats
        session.current_episode = 0
        session.total_profit = 0.0
        session.total_trades = 0
        session.win_rate = 0.0
        session.sharpe_ratio = 0.0
        session.max_drawdown = 0.0
        session.status = 'ready'
        
        db.session.commit()
        
        # Emit WebSocket event to clear trades on frontend
        socketio.emit('session_reset', {
            'session_id': session_id,
            'message': 'Session reset successfully'
        }, room=f'session_{session_id}')
        
        # Also emit to general room in case client hasn't joined session room yet
        socketio.emit('session_reset', {
            'session_id': session_id,
            'message': 'Session reset successfully'
        })
        
        return jsonify({'success': True, 'message': 'Session reset successfully'})
        
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Trading Data API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trades with optional filtering"""
    try:
        session_id = request.args.get('session_id', type=int)
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        query = Trade.query
        
        if session_id:
            query = query.filter_by(session_id=session_id)
            
        trades = query.order_by(Trade.entry_time.desc()).limit(limit).offset(offset).all()
        
        result = []
        for trade in trades:
            result.append({
                'id': trade.id,
                'session_id': trade.session_id,
                'position_type': trade.position_type,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'profit_loss': trade.profit_loss,
                'status': trade.status,
                'episode_number': trade.episode_number,
                'trade_id': trade.trade_id
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching trades: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/<int:trade_id>', methods=['GET'])
def get_trade(trade_id):
    """Get a specific trade"""
    try:
        trade = Trade.query.get(trade_id)
        if not trade:
            return jsonify({'error': 'Trade not found'}), 404
            
        return jsonify({
            'id': trade.id,
            'session_id': trade.session_id,
            'position_type': trade.position_type,
            'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
            'entry_price': trade.entry_price,
            'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'profit_loss': trade.profit_loss,
            'status': trade.status,
            'episode_number': trade.episode_number,
            'trade_id': trade.trade_id
        })
        
    except Exception as e:
        logger.error(f"Error fetching trade: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_trades', methods=['GET'])
def get_recent_trades():
    """Get recent trades across all sessions"""
    try:
        limit = request.args.get('limit', 20, type=int)
        
        trades = Trade.query.order_by(Trade.entry_time.desc()).limit(limit).all()
        
        result = []
        for trade in trades:
            result.append({
                'id': trade.id,
                'session_id': trade.session_id,
                'position_type': trade.position_type,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'profit_loss': trade.profit_loss,
                'status': trade.status,
                'episode_number': trade.episode_number
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching recent trades: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Market Data API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/market_data', methods=['GET'])
def get_market_data():
    """Get market data for charting"""
    try:
        symbol = request.args.get('symbol', 'NQ')
        timeframe = request.args.get('timeframe', '1min')
        limit = int(request.args.get('limit', 1000))
        
        data = MarketData.query.filter_by(
            symbol=symbol,
            timeframe=timeframe
        ).order_by(MarketData.timestamp.desc()).limit(limit).all()
        
        result = []
        for row in data:
            result.append({
                'timestamp': row.timestamp.isoformat(),
                'open': row.open_price,
                'high': row.high_price,
                'low': row.low_price,
                'close': row.close_price,
                'volume': row.volume
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching market data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Training Metrics API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/training_metrics/<int:session_id>', methods=['GET'])
def get_training_metrics(session_id):
    """Get training metrics for a session"""
    try:
        metrics = TrainingMetrics.query.filter_by(
            session_id=session_id
        ).order_by(TrainingMetrics.episode.desc()).limit(100).all()
        
        result = []
        for metric in metrics:
            result.append({
                'episode': metric.episode,
                'timestamp': metric.timestamp.isoformat(),
                'reward': metric.reward,
                'loss': metric.loss,
                'epsilon': metric.epsilon,
                'learning_rate': metric.learning_rate,
                'action_distribution': metric.action_distribution
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching training metrics: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Algorithm Configuration API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/algorithm_configs', methods=['GET'])
def get_algorithm_configs():
    """Get available algorithm configurations"""
    try:
        configs = AlgorithmConfig.query.filter_by(is_active=True).all()
        result = []
        
        for config in configs:
            result.append({
                'id': config.id,
                'name': config.name,
                'algorithm_type': config.algorithm_type,
                'parameters': config.parameters,
                'description': config.description
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching algorithm configs: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/algorithm_configs', methods=['POST'])
@retry_on_db_error()
def create_algorithm_config():
    """Create a new algorithm configuration"""
    try:
        data = request.get_json()
        
        config = AlgorithmConfig(
            name=data.get('name'),
            algorithm_type=data.get('algorithm_type'),
            parameters=data.get('parameters', {}),
            description=data.get('description', ''),
            is_active=data.get('is_active', True)
        )
        
        db.session.add(config)
        db.session.commit()
        
        return jsonify({
            'id': config.id,
            'name': config.name,
            'algorithm_type': config.algorithm_type,
            'parameters': config.parameters,
            'description': config.description,
            'is_active': config.is_active
        })
        
    except Exception as e:
        logger.error(f"Error creating algorithm config: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Trading Status API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/status', methods=['GET'])
def get_system_status():
    """Get overall system status"""
    try:
        active_sessions = TradingSession.query.filter_by(status='active').count()
        total_sessions = TradingSession.query.count()
        total_trades = Trade.query.count()
        
        # Get latest trades
        recent_trades = Trade.query.order_by(Trade.entry_time.desc()).limit(5).all()
        recent_trades_data = []
        for trade in recent_trades:
            recent_trades_data.append({
                'session_id': trade.session_id,
                'position_type': trade.position_type,
                'profit_loss': trade.profit_loss,
                'entry_time': trade.entry_time.isoformat() if trade.entry_time else None
            })
        
        # Get performance metrics
        active_session = TradingSession.query.filter_by(status='active').first()
        performance_metrics = {}
        if active_session:
            performance_metrics = {
                'current_episode': active_session.current_episode,
                'total_episodes': active_session.total_episodes,
                'total_profit': active_session.total_profit,
                'win_rate': active_session.win_rate,
                'sharpe_ratio': active_session.sharpe_ratio,
                'max_drawdown': active_session.max_drawdown
            }
        
        return jsonify({
            'system_status': 'online',
            'active_sessions': active_sessions,
            'total_sessions': total_sessions,
            'total_trades': total_trades,
            'recent_trades': recent_trades_data,
            'performance_metrics': performance_metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session_status/<status>', methods=['POST'])
@retry_on_db_error()
def update_session_status(status):
    """Update session status for all active sessions"""
    try:
        # Update all active sessions to the specified status
        active_sessions = TradingSession.query.filter_by(status='active').all()
        
        for session in active_sessions:
            session.status = status
            if status == 'stopped':
                session.end_time = datetime.now(timezone.utc)
                
        db.session.commit()
        
        return jsonify({
            'success': True,
            'updated_sessions': len(active_sessions),
            'new_status': status
        })
        
    except Exception as e:
        logger.error(f"Error updating session status: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Live Trading API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/place_trade', methods=['POST'])
@retry_on_db_error()
def place_trade():
    """Place a manual trade"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['symbol', 'action', 'quantity']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create trade record
        trade = Trade(
            session_id=data.get('session_id'),
            symbol=data['symbol'],
            position_type=data['action'].upper(),
            entry_time=datetime.now(timezone.utc),
            entry_price=data.get('price', 0.0),
            quantity=data['quantity'],
            notes=data.get('notes', 'Manual trade')
        )
        
        db.session.add(trade)
        db.session.commit()
        
        # Emit trade update via WebSocket
        socketio.emit('new_trade', {
            'id': trade.id,
            'symbol': trade.symbol,
            'position_type': trade.position_type,
            'entry_price': trade.entry_price,
            'quantity': trade.quantity,
            'timestamp': trade.entry_time.isoformat()
        })
        
        return jsonify({
            'success': True,
            'trade_id': trade.id,
            'message': 'Trade placed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error placing trade: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# Data Management API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/load_data', methods=['POST'])
@retry_on_db_error()
def load_market_data():
    """Load market data from file or download"""
    try:
        data = request.get_json()
        symbol = data.get('symbol', 'NQ')
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        
        # Initialize data manager
        data_manager = DataManager()
        
        # Load data
        df = data_manager.load_nq_data(start_date=start_date, end_date=end_date)
        
        if df is not None:
            return jsonify({
                'success': True,
                'rows_loaded': len(df),
                'start_date': df.index[0].isoformat() if len(df) > 0 else None,
                'end_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'message': f'Successfully loaded {len(df)} rows of market data'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to load market data'
            }), 500
            
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ---------------------------------------------------------------------------
# System Control API Endpoints
# ---------------------------------------------------------------------------

@app.route('/api/shutdown', methods=['POST'])
def shutdown_system():
    """Gracefully shutdown the system"""
    try:
        # Stop all active training sessions
        active_sessions = TradingSession.query.filter_by(status='active').all()
        for session in active_sessions:
            trading_engine.stop_training(session.id)
            session.status = 'stopped'
            session.end_time = datetime.now(timezone.utc)
            
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'System shutdown initiated',
            'stopped_sessions': len(active_sessions)
        })
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_all_sessions', methods=['POST'])
@retry_on_db_error()
def clear_all_sessions():
    """Clear all training sessions (for debugging)"""
    try:
        # Clear any stuck sessions in the trading engine
        trading_engine.clear_all_sessions()
        
        # Update database
        active_sessions = TradingSession.query.filter_by(status='active').all()
        for session in active_sessions:
            session.status = 'stopped'
            session.end_time = datetime.now(timezone.utc)
            
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'All sessions cleared',
            'cleared_count': len(active_sessions)
        })
        
    except Exception as e:
        logger.error(f"Error clearing sessions: {str(e)}")
        return jsonify({'error': str(e)}), 500