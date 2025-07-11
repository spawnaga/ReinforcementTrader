from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, trading_engine
from extensions import db, socketio
from models import TradingSession, Trade, MarketData, TrainingMetrics, AlgorithmConfig
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    active_sessions = TradingSession.query.filter_by(status='active').all()
    recent_trades = Trade.query.order_by(Trade.entry_time.desc()).limit(10).all()
    
    return render_template('index.html', 
                         active_sessions=active_sessions,
                         recent_trades=recent_trades)

@app.route('/trading_dashboard')
def trading_dashboard():
    """Advanced trading dashboard with real-time charts"""
    return render_template('trading_dashboard.html')

@app.route('/test_api')
def test_api():
    """Test API endpoint"""
    from flask import send_file
    return send_file('test_api.html')

@app.route('/strategy_builder')
def strategy_builder():
    """Interactive strategy builder interface"""
    algorithms = AlgorithmConfig.query.filter_by(is_active=True).all()
    return render_template('strategy_builder.html', algorithms=algorithms)

@app.route('/start_training', methods=['POST'])
def start_training():
    """Start a new training session"""
    try:
        data = request.get_json()
        
        # Create new trading session
        session = TradingSession(
            session_name=data.get('session_name', f'Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            algorithm_type=data.get('algorithm_type', 'ANE_PPO'),
            parameters=data.get('parameters', {}),
            total_episodes=data.get('total_episodes', 1000)
        )
        
        db.session.add(session)
        db.session.commit()
        
        # Start training in background
        trading_engine.start_training(session.id, data)
        
        return jsonify({
            'success': True,
            'session_id': session.id,
            'message': 'Training started successfully'
        })
        
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stop_training', methods=['POST'])
def stop_training():
    """Stop a training session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        session = TradingSession.query.get(session_id)
        if session:
            session.status = 'stopped'
            session.end_time = datetime.utcnow()
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

@app.route('/api/market_data')
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

@app.route('/api/training_metrics/<int:session_id>')
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

@app.route('/api/algorithm_configs')
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

@app.route('/api/sessions')
def get_sessions():
    """Get all trading sessions"""
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

@app.route('/api/session_status/<int:session_id>')
def get_session_status(session_id):
    """Get real-time status of a training session"""
    try:
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Get recent trades
        recent_trades = Trade.query.filter_by(session_id=session_id).order_by(Trade.entry_time.desc()).limit(10).all()
        
        trades_data = []
        for trade in recent_trades:
            trades_data.append({
                'trade_id': trade.trade_id,
                'entry_time': trade.entry_time.isoformat(),
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position_type': trade.position_type,
                'profit_loss': trade.profit_loss,
                'status': trade.status
            })
        
        return jsonify({
            'session': {
                'id': session.id,
                'name': session.session_name,
                'status': session.status,
                'current_episode': session.current_episode,
                'total_episodes': session.total_episodes,
                'total_profit': session.total_profit,
                'total_trades': session.total_trades,
                'win_rate': session.win_rate,
                'sharpe_ratio': session.sharpe_ratio,
                'max_drawdown': session.max_drawdown
            },
            'recent_trades': trades_data
        })
        
    except Exception as e:
        logger.error(f"Error fetching session status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/interactive_brokers/connect', methods=['POST'])
def connect_interactive_brokers():
    """Connect to Interactive Brokers"""
    try:
        data = request.get_json()
        host = data.get('host', '127.0.0.1')
        port = data.get('port', 7497)
        client_id = data.get('client_id', 1)
        
        success = trading_engine.connect_ib(host, port, client_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Connected to Interactive Brokers successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to connect to Interactive Brokers'
            }), 500
            
    except Exception as e:
        logger.error(f"Error connecting to IB: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/interactive_brokers/disconnect', methods=['POST'])
def disconnect_interactive_brokers():
    """Disconnect from Interactive Brokers"""
    try:
        trading_engine.disconnect_ib()
        return jsonify({
            'success': True,
            'message': 'Disconnected from Interactive Brokers'
        })
        
    except Exception as e:
        logger.error(f"Error disconnecting from IB: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
