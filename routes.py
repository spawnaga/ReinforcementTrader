from flask import render_template, request, jsonify, redirect, url_for, flash
from app import app, trading_engine
from extensions import db, socketio
from models import TradingSession, Trade, MarketData, TrainingMetrics, AlgorithmConfig
from datetime import datetime, timezone
import json
import logging
from db_utils import retry_on_db_error
import pandas as pd

# Import websocket handlers to register them
import websocket_handler

logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Main dashboard page"""
    active_sessions = TradingSession.query.filter_by(status='active').all()
    recent_trades = Trade.query.order_by(Trade.entry_time.desc()).limit(10).all()
    total_trades_count = Trade.query.count()
    
    return render_template('index.html', 
                         active_sessions=active_sessions,
                         recent_trades=recent_trades,
                         total_trades_count=total_trades_count)

@app.route('/test_chart')
def test_chart():
    """Test chart page for debugging"""
    return render_template('test_chart.html')

@app.route('/test_chart_simple')
def test_chart_simple():
    """Simple inline chart test"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Chart Test</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <h1>Simple Chart Test</h1>
        <canvas id="myChart" width="400" height="200"></canvas>
        <script>
            const ctx = document.getElementById('myChart').getContext('2d');
            const myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
                    datasets: [{
                        label: '# of Votes',
                        data: [12, 19, 3, 5, 2, 3],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                }
            });
            console.log('Chart created:', myChart);
        </script>
    </body>
    </html>
    '''

@app.route('/trading_dashboard')
def trading_dashboard():
    """Advanced trading dashboard with real-time charts"""
    return render_template('trading_dashboard.html')

@app.route('/enhanced_dashboard')
def enhanced_dashboard():
    """Enhanced trading dashboard with full neural network control"""
    return render_template('enhanced_trading_dashboard.html')

@app.route('/test_trades')
def test_trades():
    """Test page for API endpoints"""
    return render_template('test_trades.html')

@app.route('/api/debug_trades')
def debug_trades():
    """Debug endpoint to check trades"""
    try:
        trades = Trade.query.limit(5).all()
        result = {
            'count': len(trades),
            'trades': []
        }
        for trade in trades:
            result['trades'].append({
                'id': trade.id,
                'session_id': trade.session_id,
                'entry_price': trade.entry_price,
                'position_type': trade.position_type
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@app.route('/strategy_builder')
def strategy_builder():
    """Interactive strategy builder interface"""
    algorithms = AlgorithmConfig.query.filter_by(is_active=True).all()
    return render_template('strategy_builder.html', algorithms=algorithms)

@app.route('/start_training', methods=['POST'])
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

@app.route('/test_training', methods=['GET'])
@retry_on_db_error()
def test_training():
    """Test training with default parameters"""
    try:
        logger.info("Testing training system")
        
        # Create test session
        test_data = {
            'session_name': f'Test_Training_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'algorithm_type': 'ANE_PPO',
            'total_episodes': 10,  # Small number for testing
            'parameters': {
                'learning_rate': 3e-4,
                'gamma': 0.99
            }
        }
        
        # Create session
        session = TradingSession(
            session_name=test_data['session_name'],
            algorithm_type=test_data['algorithm_type'],
            parameters=test_data['parameters'],
            total_episodes=test_data['total_episodes']
        )
        
        db.session.add(session)
        db.session.commit()
        
        # Start training
        success = trading_engine.start_training(session.id, test_data)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Test training started! Session ID: {session.id}. Check the "Active Training Sessions" section below for progress.',
                'session_id': session.id
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Another training session is already running. Please wait for it to complete before starting a new one.'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in test training: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stop_training', methods=['POST'])
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

@app.route('/api/sessions/<int:session_id>/start', methods=['POST'])
@retry_on_db_error()
def start_session(session_id):
    """Start a training session"""
    try:
        session = TradingSession.query.get(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
            
        # Start training with the session parameters
        success = trading_engine.start_training(session_id, session.configuration)
        
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
        from extensions import socketio
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
                configuration=data.get('parameters', {})
            )
            
            db.session.add(new_session)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'session_id': new_session.id,
                'message': 'New training session created'
            })
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/session_status/<status>')
def get_sessions_by_status(status):
    """Get trading sessions by status"""
    try:
        sessions = TradingSession.query.filter_by(status=status).order_by(TradingSession.start_time.desc()).all()
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
                'total_profit': session.total_profit
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching sessions by status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_trades')
def get_recent_trades():
    """Get recent trades for a session or all sessions"""
    session_id = request.args.get('session_id', type=int)
    limit = request.args.get('limit', 10, type=int)
    
    try:
        query = Trade.query
        if session_id:
            query = query.filter_by(session_id=session_id)
        
        # For now, show all trades if no specific session is requested
        # This ensures the dashboard displays data even if trades are from old sessions
        trades = query.order_by(Trade.entry_time.desc()).limit(limit).all()
        
        logger.debug(f"Found {len(trades)} trades")
        
        trades_data = []
        for trade in trades:
            if trade:  # Add null check
                trades_data.append({
                    'id': trade.id,
                    'position_type': trade.position_type,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'profit_loss': trade.profit_loss,
                    'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                    'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                    'quantity': trade.quantity,
                    'session_id': trade.session_id
                })
        
        logger.debug(f"Returning {len(trades_data)} trades")
        return jsonify(trades_data), 200
    except Exception as e:
        logger.error(f"Error fetching recent trades: {str(e)}")
        logger.error(f"Full error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to fetch recent trades'}), 500

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


@app.route("/api/system_test")
def system_test():
    """Test the training system setup"""
    try:
        results = {
            "database": False,
            "data_loading": False,
            "state_creation": False,
            "environment": False,
            "algorithm": False,
            "details": {}
        }
        
        # Test 1: Database connection
        from models import MarketData
        try:
            count = MarketData.query.count()
            results["database"] = True
            results["details"]["market_data_count"] = count
        except Exception as e:
            results["details"]["database_error"] = str(e)
        
        # Test 2: Data loading
        data = None
        try:
            # Use load_futures_data instead of load_nq_data
            data = data_manager.load_futures_data('NQ')
            if data is not None:
                results["data_loading"] = True
                results["details"]["data_rows"] = len(data)
            else:
                results["details"]["data_error"] = "No data loaded"
        except Exception as e:
            results["details"]["data_error"] = str(e)
        
        # Test 3: State creation
        state = None
        if results["data_loading"] and data is not None:
            try:
                test_data = data.head(100)
                # Add time column
                if "time" not in test_data.columns:
                    if test_data.index.name == "timestamp":
                        test_data = test_data.reset_index()
                        test_data.rename(columns={"timestamp": "time"}, inplace=True)
                    else:
                        test_data["time"] = pd.date_range(start="2025-01-01", periods=len(test_data), freq="1min")
                
                from gym_futures.envs.utils import TimeSeriesState
                state = TimeSeriesState(test_data)
                results["state_creation"] = True
                results["details"]["state_created"] = True
            except Exception as e:
                results["details"]["state_error"] = str(e)
        
        # Test 4: Environment
        env = None
        if results["state_creation"] and state is not None:
            try:
                from gym_futures.envs.futures_env import FuturesEnv
                env = FuturesEnv(states=[state], value_per_tick=5.0, tick_size=0.25)
                results["environment"] = True
                results["details"]["env_created"] = True
            except Exception as e:
                results["details"]["env_error"] = str(e)
        
        # Test 5: Algorithm
        if results["environment"] and env is not None:
            try:
                from rl_algorithms.ane_ppo import ANEPPO
                import torch
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                algo = ANEPPO(env=env, device=device)
                results["algorithm"] = True
                results["details"]["device"] = str(device)
            except Exception as e:
                results["details"]["algorithm_error"] = str(e)
        
        # Overall status
        all_passed = all([
            results["database"], 
            results["data_loading"], 
            results["state_creation"],
            results["environment"],
            results["algorithm"]
        ])
        
        return jsonify({
            "success": all_passed,
            "tests": results,
            "message": "All systems operational!" if all_passed else "Some tests failed"
        })
        
    except Exception as e:
        logger.error(f"System test error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/validate_data', methods=['POST'])
def validate_data():
    """Validate data selection and settings"""
    try:
        data = request.json
        range_type = data.get('type', 'all')
        
        # Load data based on selection
        data_manager = DataManager()
        market_data = None
        
        if range_type == 'percentage':
            percentage = data.get('percentage', 100)
            market_data = data_manager.load_nq_data()
            if market_data is not None:
                total_rows = len(market_data)
                rows_to_use = int(total_rows * percentage / 100)
                market_data = market_data.tail(rows_to_use)
                
        elif range_type == 'daterange':
            start_date = data.get('startDate')
            end_date = data.get('endDate')
            market_data = data_manager.load_nq_data(start_date=start_date, end_date=end_date)
            
        elif range_type == 'timeperiod':
            period = data.get('period', 'all')
            if period != 'all':
                # Calculate date range based on period
                from datetime import datetime, timedelta
                end_date = datetime.now()
                
                period_map = {
                    '1month': 30,
                    '3months': 90,
                    '6months': 180,
                    '1year': 365,
                    '2years': 730,
                    '3years': 1095,
                    '5years': 1825,
                    '10years': 3650
                }
                
                days = period_map.get(period, 365)
                start_date = end_date - timedelta(days=days)
                market_data = data_manager.load_nq_data(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
            else:
                market_data = data_manager.load_nq_data()
        
        if market_data is not None and len(market_data) > 0:
            return jsonify({
                'valid': True,
                'rowCount': len(market_data),
                'startDate': str(market_data.index[0]) if hasattr(market_data.index[0], 'date') else str(market_data.index[0]),
                'endDate': str(market_data.index[-1]) if hasattr(market_data.index[-1], 'date') else str(market_data.index[-1])
            })
        else:
            return jsonify({
                'valid': False,
                'error': 'No data available for the selected range'
            })
            
    except Exception as e:
        logger.error(f"Data validation error: {str(e)}")
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500

@app.route('/chart_debug')
def chart_debug():
    """Chart debug page"""
    return render_template('chart_debug.html')


