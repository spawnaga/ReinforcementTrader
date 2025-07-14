#!/usr/bin/env python3
"""Monitor for infinite trading patterns"""

import time
import logging
from datetime import datetime, timedelta
from models import Trade, TradingSession
from app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_trading_patterns():
    """Monitor for suspicious trading patterns"""
    
    with app.app_context():
        while True:
            # Check active sessions
            active_sessions = TradingSession.query.filter_by(status='active').all()
            
            for session in active_sessions:
                # Check trades in last minute
                one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
                recent_trades = Trade.query.filter(
                    Trade.session_id == session.id,
                    Trade.timestamp > one_minute_ago
                ).count()
                
                if recent_trades > 10:
                    logger.warning(f"Session {session.id}: {recent_trades} trades in last minute!")
                    
                # Check for same timestamp trades
                same_time_trades = db.session.query(
                    Trade.timestamp, 
                    func.count(Trade.id).label('count')
                ).filter(
                    Trade.session_id == session.id
                ).group_by(Trade.timestamp).having(
                    func.count(Trade.id) > 5
                ).all()
                
                if same_time_trades:
                    logger.error(f"Session {session.id} has multiple trades at same timestamp!")
                    for ts, count in same_time_trades:
                        logger.error(f"  {ts}: {count} trades")
            
            time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    monitor_trading_patterns()
