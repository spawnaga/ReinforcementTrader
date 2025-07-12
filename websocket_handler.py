import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from flask import request
from flask_socketio import emit, join_room, leave_room, disconnect
from app import app, trading_engine
from extensions import socketio, db
from models import TradingSession, TrainingMetrics, Trade
import threading
import time

logger = logging.getLogger(__name__)

# Active connections tracking
active_connections = {}
room_subscriptions = {}

# Real-time data broadcasting
broadcast_thread = None
broadcast_active = False

@socketio.on('connect')
def handle_connect(auth=None):
    """Handle client connection"""
    try:
        client_id = request.sid
        active_connections[client_id] = {
            'connected_at': datetime.utcnow(),
            'subscriptions': set(),
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr
        }
        
        logger.info(f"Client {client_id} connected from {request.remote_addr}")
        
        # Send initial connection confirmation
        emit('connection_confirmed', {
            'client_id': client_id,
            'server_time': datetime.utcnow().isoformat(),
            'message': 'Connected to AI Trading System'
        })
        
        # Send system status
        emit('system_status', get_system_status())
        
        # Start global broadcast thread if not already running
        start_broadcast_thread()
        
    except Exception as e:
        logger.error(f"Error handling connection: {str(e)}")
        emit('error', {'message': 'Connection error'})

@socketio.on('disconnect')
def handle_disconnect(*args, **kwargs):
    """Handle client disconnection"""
    try:
        client_id = request.sid
        
        if client_id in active_connections:
            # Remove from all rooms
            subscriptions = active_connections[client_id].get('subscriptions', set())
            for room in subscriptions:
                leave_room(room)
                if room in room_subscriptions:
                    room_subscriptions[room].discard(client_id)
                    if not room_subscriptions[room]:
                        del room_subscriptions[room]
            
            del active_connections[client_id]
            
        logger.info(f"Client {client_id} disconnected")
        
        # Stop broadcast thread if no more connections
        if not active_connections:
            stop_broadcast_thread()
            
    except Exception as e:
        logger.error(f"Error handling disconnection: {str(e)}")

@socketio.on('subscribe_session')
def handle_subscribe_session(data):
    """Subscribe to training session updates"""
    try:
        client_id = request.sid
        session_id = data.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        # Verify session exists
        with app.app_context():
            session = TradingSession.query.get(session_id)
            if not session:
                emit('error', {'message': 'Session not found'})
                return
        
        room_name = f"session_{session_id}"
        join_room(room_name)
        
        # Track subscription
        if client_id in active_connections:
            active_connections[client_id]['subscriptions'].add(room_name)
        
        if room_name not in room_subscriptions:
            room_subscriptions[room_name] = set()
        room_subscriptions[room_name].add(client_id)
        
        logger.info(f"Client {client_id} subscribed to session {session_id}")
        
        # Send current session status
        emit('session_status', {
            'session_id': session_id,
            'status': session.status,
            'current_episode': session.current_episode,
            'total_episodes': session.total_episodes,
            'total_profit': session.total_profit
        })
        
    except Exception as e:
        logger.error(f"Error subscribing to session: {str(e)}")
        emit('error', {'message': 'Subscription error'})

@socketio.on('unsubscribe_session')
def handle_unsubscribe_session(data):
    """Unsubscribe from training session updates"""
    try:
        client_id = request.sid
        session_id = data.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        room_name = f"session_{session_id}"
        leave_room(room_name)
        
        # Remove subscription tracking
        if client_id in active_connections:
            active_connections[client_id]['subscriptions'].discard(room_name)
        
        if room_name in room_subscriptions:
            room_subscriptions[room_name].discard(client_id)
            if not room_subscriptions[room_name]:
                del room_subscriptions[room_name]
        
        logger.info(f"Client {client_id} unsubscribed from session {session_id}")
        
        emit('unsubscribed', {'session_id': session_id})
        
    except Exception as e:
        logger.error(f"Error unsubscribing from session: {str(e)}")
        emit('error', {'message': 'Unsubscription error'})

@socketio.on('start_ai_analysis')
def handle_start_ai_analysis(data):
    """Start AI analysis for a session"""
    try:
        session_id = data.get('session_id')
        risk_level = data.get('risk_level', 'moderate')
        position_size = data.get('position_size', 1)
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        # Start AI analysis through trading engine
        success = trading_engine.start_ai_analysis(session_id, {
            'risk_level': risk_level,
            'position_size': position_size,
            'client_id': request.sid
        })
        
        if success:
            emit('ai_analysis_started', {
                'session_id': session_id,
                'message': 'AI analysis started successfully'
            })
            
            # Broadcast to session room
            socketio.emit('ai_status_update', {
                'session_id': session_id,
                'status': 'active',
                'risk_level': risk_level
            }, room=f"session_{session_id}")
            
        else:
            emit('error', {'message': 'Failed to start AI analysis'})
            
    except Exception as e:
        logger.error(f"Error starting AI analysis: {str(e)}")
        emit('error', {'message': 'AI analysis error'})

@socketio.on('stop_ai_analysis')
def handle_stop_ai_analysis():
    """Stop AI analysis"""
    try:
        client_id = request.sid
        
        # Stop AI analysis through trading engine
        success = trading_engine.stop_ai_analysis(client_id)
        
        if success:
            emit('ai_analysis_stopped', {
                'message': 'AI analysis stopped successfully'
            })
        else:
            emit('error', {'message': 'Failed to stop AI analysis'})
            
    except Exception as e:
        logger.error(f"Error stopping AI analysis: {str(e)}")
        emit('error', {'message': 'AI analysis stop error'})

@socketio.on('request_ai_update')
def handle_request_ai_update(data):
    """Request AI update for current market conditions"""
    try:
        session_id = data.get('session_id')
        current_price = data.get('current_price')
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        # Get AI recommendation from trading engine
        recommendation = trading_engine.get_ai_recommendation(session_id, current_price)
        
        if recommendation:
            emit('ai_recommendation', {
                'session_id': session_id,
                'action': recommendation.get('action'),
                'confidence': recommendation.get('confidence'),
                'reasoning': recommendation.get('reasoning'),
                'timestamp': datetime.utcnow().isoformat()
            })
        
    except Exception as e:
        logger.error(f"Error requesting AI update: {str(e)}")
        emit('error', {'message': 'AI update error'})

@socketio.on('request_real_time_update')
def handle_request_real_time_update(data):
    """Request real-time updates for a session"""
    try:
        session_id = data.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        # Get current session data
        session_data = get_session_real_time_data(session_id)
        
        if session_data:
            emit('real_time_update', session_data)
        
    except Exception as e:
        logger.error(f"Error requesting real-time update: {str(e)}")
        emit('error', {'message': 'Real-time update error'})

@socketio.on('request_system_status')
def handle_request_system_status():
    """Request current system status"""
    try:
        status = get_system_status()
        emit('system_status', status)
        
    except Exception as e:
        logger.error(f"Error requesting system status: {str(e)}")
        emit('error', {'message': 'System status error'})

@socketio.on('subscribe_market_data')
def handle_subscribe_market_data(data):
    """Subscribe to market data updates"""
    try:
        client_id = request.sid
        symbol = data.get('symbol', 'NQ')
        timeframe = data.get('timeframe', '1min')
        
        room_name = f"market_data_{symbol}_{timeframe}"
        join_room(room_name)
        
        # Track subscription
        if client_id in active_connections:
            active_connections[client_id]['subscriptions'].add(room_name)
        
        if room_name not in room_subscriptions:
            room_subscriptions[room_name] = set()
        room_subscriptions[room_name].add(client_id)
        
        logger.info(f"Client {client_id} subscribed to market data {symbol} {timeframe}")
        
        emit('market_data_subscribed', {
            'symbol': symbol,
            'timeframe': timeframe
        })
        
    except Exception as e:
        logger.error(f"Error subscribing to market data: {str(e)}")
        emit('error', {'message': 'Market data subscription error'})

@socketio.on('place_manual_trade')
def handle_place_manual_trade(data):
    """Handle manual trade placement"""
    try:
        action = data.get('action')
        symbol = data.get('symbol', 'NQ')
        position_size = data.get('position_size', 1)
        price = data.get('price')
        
        if not action or action not in ['buy', 'sell', 'hold']:
            emit('error', {'message': 'Invalid action'})
            return
        
        # Place trade through trading engine
        result = trading_engine.place_manual_trade({
            'action': action,
            'symbol': symbol,
            'position_size': position_size,
            'price': price,
            'client_id': request.sid
        })
        
        if result.get('success'):
            emit('trade_executed', {
                'trade_id': result.get('trade_id'),
                'action': action,
                'symbol': symbol,
                'position_size': position_size,
                'price': result.get('executed_price'),
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            emit('trade_rejected', {
                'reason': result.get('error'),
                'action': action
            })
            
    except Exception as e:
        logger.error(f"Error placing manual trade: {str(e)}")
        emit('error', {'message': 'Trade placement error'})

@socketio.on('request_performance_metrics')
def handle_request_performance_metrics(data):
    """Request performance metrics for a session"""
    try:
        session_id = data.get('session_id')
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        metrics = get_performance_metrics(session_id)
        emit('performance_metrics', metrics)
        
    except Exception as e:
        logger.error(f"Error requesting performance metrics: {str(e)}")
        emit('error', {'message': 'Performance metrics error'})

@socketio.on('update_risk_parameters')
def handle_update_risk_parameters(data):
    """Update risk management parameters"""
    try:
        session_id = data.get('session_id')
        risk_params = data.get('risk_parameters', {})
        
        if not session_id:
            emit('error', {'message': 'Session ID required'})
            return
        
        # Update risk parameters through trading engine
        success = trading_engine.update_risk_parameters(session_id, risk_params)
        
        if success:
            emit('risk_parameters_updated', {
                'session_id': session_id,
                'parameters': risk_params
            })
            
            # Broadcast to session room
            socketio.emit('risk_update', {
                'session_id': session_id,
                'parameters': risk_params
            }, room=f"session_{session_id}")
            
        else:
            emit('error', {'message': 'Failed to update risk parameters'})
            
    except Exception as e:
        logger.error(f"Error updating risk parameters: {str(e)}")
        emit('error', {'message': 'Risk parameter update error'})

# Utility functions

def get_system_status() -> Dict[str, Any]:
    """Get current system status"""
    try:
        return {
            'server_time': datetime.utcnow().isoformat(),
            'active_connections': len(active_connections),
            'active_sessions': len(trading_engine.get_active_sessions()),
            'gpu_available': trading_engine.device.type == 'cuda' if hasattr(trading_engine, 'device') else False,
            'ib_connected': hasattr(trading_engine, 'ib_integration') and trading_engine.ib_integration.connected,
            'market_open': trading_engine.is_market_open() if hasattr(trading_engine, 'is_market_open') else False,
            'system_health': 'healthy'
        }
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return {'error': 'Failed to get system status'}

def get_session_real_time_data(session_id: int) -> Optional[Dict[str, Any]]:
    """Get real-time data for a training session with retry logic for database locks"""
    import time
    import random
    from sqlalchemy.exc import OperationalError
    from sqlalchemy import text
    
    max_retries = 5  # Increase retries
    base_delay = 0.5  # Start with 500ms
    
    for attempt in range(max_retries):
        try:
            with app.app_context():
                # Use a read-only transaction with shorter timeout
                with db.session.begin():
                    # Set SQLite to read-only mode for this query
                    if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']:
                        db.session.execute(text("PRAGMA query_only = ON"))
                    
                    session = TradingSession.query.get(session_id)
                    if not session:
                        return None
                    
                    # Store basic session data
                    session_data = {
                        'session_id': session_id,
                        'current_episode': session.current_episode,
                        'total_episodes': session.total_episodes,
                        'total_profit': session.total_profit,
                        'total_trades': session.total_trades,
                        'win_rate': session.win_rate
                    }
                    
                    # Get latest training metrics with a simpler query
                    latest_metrics = db.session.execute(
                        text("""
                            SELECT episode, reward, loss, timestamp 
                            FROM training_metrics 
                            WHERE session_id = :session_id 
                            ORDER BY episode DESC 
                            LIMIT 1
                        """),
                        {'session_id': session_id}
                    ).first()
                    
                    if latest_metrics:
                        session_data['latest_metrics'] = {
                            'episode': latest_metrics[0],
                            'reward': latest_metrics[1],
                            'loss': latest_metrics[2],
                            'timestamp': latest_metrics[3].isoformat() if latest_metrics[3] else None
                        }
                    else:
                        session_data['latest_metrics'] = None
                    
                    # Get recent trades with a simpler query
                    recent_trades = db.session.execute(
                        text("""
                            SELECT trade_id, entry_time, position_type, profit_loss, status 
                            FROM trade 
                            WHERE session_id = :session_id 
                            ORDER BY entry_time DESC 
                            LIMIT 10
                        """),
                        {'session_id': session_id}
                    ).fetchall()
                    
                    session_data['recent_trades'] = [{
                        'trade_id': trade[0],
                        'entry_time': trade[1].isoformat() if trade[1] else None,
                        'position_type': trade[2],
                        'profit_loss': trade[3],
                        'status': trade[4]
                    } for trade in recent_trades]
                    
                    # Reset query_only mode
                    if 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']:
                        db.session.execute(text("PRAGMA query_only = OFF"))
                    
                    return session_data
                
        except OperationalError as e:
            if 'database is locked' in str(e) and attempt < max_retries - 1:
                # Add random jitter to prevent thundering herd
                jitter = random.uniform(0, 0.3)
                sleep_time = base_delay + jitter
                logger.warning(f"Database locked, retrying in {sleep_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(sleep_time)
                base_delay *= 1.5  # Less aggressive exponential backoff
                continue
            else:
                logger.error(f"Error getting session real-time data after {attempt + 1} attempts: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error getting session real-time data: {str(e)}")
            return None

def get_performance_metrics(session_id: int) -> Dict[str, Any]:
    """Get performance metrics for a session"""
    try:
        with app.app_context():
            session = TradingSession.query.get(session_id)
            if not session:
                return {'error': 'Session not found'}
            
            # Calculate additional metrics
            total_trades = Trade.query.filter_by(session_id=session_id, status='closed').count()
            profitable_trades = Trade.query.filter_by(session_id=session_id, status='closed').filter(Trade.profit_loss > 0).count()
            
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Get profit/loss history
            trades = Trade.query.filter_by(session_id=session_id, status='closed').order_by(Trade.exit_time).all()
            pnl_history = [trade.profit_loss for trade in trades]
            
            # Calculate drawdown
            cumulative_pnl = []
            running_total = 0
            for pnl in pnl_history:
                running_total += pnl
                cumulative_pnl.append(running_total)
            
            max_drawdown = 0
            peak = 0
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak if peak > 0 else 0
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            # Calculate Sharpe ratio (simplified)
            if len(pnl_history) > 1:
                avg_return = sum(pnl_history) / len(pnl_history)
                volatility = (sum([(x - avg_return) ** 2 for x in pnl_history]) / (len(pnl_history) - 1)) ** 0.5
                sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            else:
                sharpe_ratio = 0
            
            return {
                'session_id': session_id,
                'total_return': session.total_profit,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'win_rate': win_rate,
                'max_drawdown': max_drawdown * 100,
                'sharpe_ratio': sharpe_ratio,
                'average_trade': sum(pnl_history) / len(pnl_history) if pnl_history else 0,
                'largest_win': max(pnl_history) if pnl_history else 0,
                'largest_loss': min(pnl_history) if pnl_history else 0,
                'profit_factor': sum([p for p in pnl_history if p > 0]) / abs(sum([p for p in pnl_history if p < 0])) if any(p < 0 for p in pnl_history) else float('inf')
            }
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        return {'error': 'Failed to get performance metrics'}

def broadcast_training_update(session_id: int, data: Dict[str, Any]):
    """Broadcast training update to subscribed clients"""
    try:
        room_name = f"session_{session_id}"
        socketio.emit('training_update', {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }, room=room_name)
        
    except Exception as e:
        logger.error(f"Error broadcasting training update: {str(e)}")

def broadcast_market_data(symbol: str, timeframe: str, data: Dict[str, Any]):
    """Broadcast market data update to subscribed clients"""
    try:
        room_name = f"market_data_{symbol}_{timeframe}"
        socketio.emit('market_data', {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': datetime.utcnow().isoformat(),
            **data
        }, room=room_name)
        
    except Exception as e:
        logger.error(f"Error broadcasting market data: {str(e)}")

def broadcast_trade_execution(trade_data: Dict[str, Any]):
    """Broadcast trade execution to all connected clients"""
    try:
        socketio.emit('trade_execution', {
            'timestamp': datetime.utcnow().isoformat(),
            **trade_data
        })
        
    except Exception as e:
        logger.error(f"Error broadcasting trade execution: {str(e)}")

def broadcast_risk_alert(session_id: int, alert_data: Dict[str, Any]):
    """Broadcast risk alert to session subscribers"""
    try:
        room_name = f"session_{session_id}"
        socketio.emit('risk_alert', {
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat(),
            **alert_data
        }, room=room_name)
        
    except Exception as e:
        logger.error(f"Error broadcasting risk alert: {str(e)}")

def broadcast_system_notification(message: str, notification_type: str = 'info'):
    """Broadcast system notification to all connected clients"""
    try:
        socketio.emit('system_notification', {
            'message': message,
            'type': notification_type,
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error broadcasting system notification: {str(e)}")

# Background broadcast thread
def start_broadcast_thread():
    """Start the background broadcast thread"""
    global broadcast_thread, broadcast_active
    
    if broadcast_thread is None or not broadcast_thread.is_alive():
        broadcast_active = True
        broadcast_thread = threading.Thread(target=background_broadcast_loop, daemon=True)
        broadcast_thread.start()
        logger.info("Background broadcast thread started")

def stop_broadcast_thread():
    """Stop the background broadcast thread"""
    global broadcast_active
    broadcast_active = False
    logger.info("Background broadcast thread stopped")

def background_broadcast_loop():
    """Background loop for periodic broadcasts"""
    global broadcast_active
    
    logger.info("Background broadcast loop started successfully")
    loop_iteration = 0
    
    while broadcast_active:
        try:
            loop_iteration += 1
            logger.debug(f"Broadcast loop iteration {loop_iteration} starting...")
            
            if active_connections:
                logger.debug(f"Active connections found: {len(active_connections)}")
                
                # Collect system performance metrics
                logger.debug("Collecting system performance metrics...")
                cpu_usage = get_cpu_usage()
                memory_usage = get_memory_usage()
                gpu_usage = get_gpu_usage()
                network_io = get_network_io()
                
                performance_data = {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'gpu_usage': gpu_usage,
                    'network_io': network_io
                }
                
                logger.info(f"Performance metrics collected - CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%, GPU: {gpu_usage:.1f}%, Network: {network_io:.2f} MB/s")
                
                # Broadcast the metrics
                logger.debug("Broadcasting performance metrics...")
                socketio.emit('performance_metrics', performance_data)
                logger.debug("Performance metrics broadcast complete")
                
                # Broadcast active session updates
                active_sessions = trading_engine.get_active_sessions()
                logger.debug(f"Found {len(active_sessions)} active training sessions")
                
                for session_id in active_sessions:
                    logger.debug(f"Getting real-time data for session {session_id}")
                    session_data = get_session_real_time_data(session_id)
                    if session_data:
                        room_name = f"session_{session_id}"
                        logger.debug(f"Broadcasting update for session {session_id}")
                        socketio.emit('session_update', session_data, room=room_name)
            else:
                logger.debug("No active connections, skipping broadcast")
            
            logger.debug(f"Broadcast loop iteration {loop_iteration} complete, sleeping for 5 seconds")
            time.sleep(5)  # Broadcast every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in background broadcast loop (iteration {loop_iteration}): {str(e)}", exc_info=True)
            time.sleep(5)
    
    logger.info("Background broadcast loop exited")

# System monitoring functions
def get_cpu_usage() -> float:
    """Get CPU usage percentage"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.debug(f"CPU usage collected: {cpu_percent}%")
        return cpu_percent
    except ImportError as e:
        logger.error(f"psutil not imported for CPU monitoring: {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting CPU usage: {str(e)}")
        return 0.0

def get_memory_usage() -> float:
    """Get memory usage percentage"""
    try:
        import psutil
        memory_percent = psutil.virtual_memory().percent
        logger.debug(f"Memory usage collected: {memory_percent}%")
        return memory_percent
    except ImportError as e:
        logger.error(f"psutil not imported for memory monitoring: {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting memory usage: {str(e)}")
        return 0.0

def get_gpu_usage() -> float:
    """Get GPU usage percentage"""
    try:
        import torch
        if torch.cuda.is_available():
            # torch.cuda.utilization() returns percentage 0-100
            gpu_util = torch.cuda.utilization()
            logger.debug(f"GPU usage collected: {gpu_util}%")
            return gpu_util
        else:
            logger.debug("No CUDA GPUs available")
            return 0.0
    except ImportError as e:
        logger.debug(f"torch not imported for GPU monitoring: {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting GPU usage: {str(e)}")
        return 0.0

def get_network_io() -> float:
    """Get network I/O rate"""
    try:
        import psutil
        net_io = psutil.net_io_counters()
        # Calculate MB/s (we'll need to track previous values for rate)
        total_bytes = net_io.bytes_sent + net_io.bytes_recv
        network_rate = total_bytes / 1024 / 1024  # Convert to MB
        logger.debug(f"Network I/O collected: {network_rate:.2f} MB total")
        return network_rate
    except ImportError as e:
        logger.error(f"psutil not imported for network monitoring: {str(e)}")
        return 0.0
    except Exception as e:
        logger.error(f"Error getting network I/O: {str(e)}")
        return 0.0

# Error handlers
@socketio.on_error()
def error_handler(e):
    """Handle WebSocket errors"""
    logger.error(f"WebSocket error: {str(e)}")
    emit('error', {'message': 'An error occurred'})

@socketio.on_error_default
def default_error_handler(e):
    """Handle default WebSocket errors"""
    logger.error(f"Default WebSocket error: {str(e)}")
    emit('error', {'message': 'An unexpected error occurred'})

# Cleanup function
def cleanup_connections():
    """Cleanup inactive connections"""
    try:
        current_time = datetime.utcnow()
        inactive_connections = []
        
        for client_id, connection_data in active_connections.items():
            # Check if connection is older than 1 hour without activity
            if (current_time - connection_data['connected_at']).total_seconds() > 3600:
                inactive_connections.append(client_id)
        
        for client_id in inactive_connections:
            if client_id in active_connections:
                del active_connections[client_id]
                logger.info(f"Cleaned up inactive connection: {client_id}")
                
    except Exception as e:
        logger.error(f"Error cleaning up connections: {str(e)}")

# Schedule periodic cleanup
import atexit
atexit.register(stop_broadcast_thread)

logger.info("WebSocket handler initialized")
