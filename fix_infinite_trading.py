#!/usr/bin/env python3
"""
Emergency fix for infinite trading loop issue

The AI is exploiting the environment by making hundreds of trades at the same timestamp.
This script will:
1. Stop the current training session that's stuck
2. Clear the excessive trades
3. Implement a fix to prevent this exploitation
"""

import os
import sys
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from models import TradingSession, Trade
from extensions import db
from app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_infinite_trading():
    """Fix the infinite trading issue"""
    
    with app.app_context():
        try:
            # First, stop any active sessions that have excessive trades
            logger.info("Checking for sessions with excessive trades...")
            
            sessions = TradingSession.query.filter_by(status='active').all()
            for session in sessions:
                trade_count = Trade.query.filter_by(session_id=session.id).count()
                if trade_count > 100:  # More than 100 trades is suspicious
                    logger.warning(f"Session {session.id} has {trade_count} trades - stopping it")
                    session.status = 'stopped'
                    session.end_time = datetime.utcnow()
                    db.session.commit()
                    
                    # Optionally clean up excessive trades (keep first 100)
                    if trade_count > 1000:
                        logger.info(f"Cleaning up excessive trades for session {session.id}")
                        # Delete all but the first 100 trades
                        excess_trades = Trade.query.filter_by(session_id=session.id)\
                            .order_by(Trade.timestamp.desc())\
                            .limit(trade_count - 100).all()
                        for trade in excess_trades:
                            db.session.delete(trade)
                        db.session.commit()
            
            # Now let's create a patched environment file
            logger.info("Creating patched environment to prevent exploitation...")
            
            patch_content = '''"""
Patch for RealisticFuturesEnv to prevent infinite trading exploitation

This patch adds:
1. State advancement tracking to prevent trading on the same state multiple times
2. Episode-wide timestamp tracking to prevent time manipulation
3. Stricter trade frequency limits
"""

def apply_infinite_trading_patch():
    """Apply patches to prevent infinite trading"""
    
    # Add this to RealisticFuturesEnv.__init__:
    # self.states_traded = set()  # Track which states have been traded
    # self.last_trade_index = -10  # Ensure minimum gap between trades
    
    # Add this check to buy() and sell() methods:
    # if self.current_index in self.states_traded:
    #     return  # Already traded at this state
    # if self.current_index - self.last_trade_index < 5:
    #     return  # Too soon since last trade
    
    # After successful trade:
    # self.states_traded.add(self.current_index)
    # self.last_trade_index = self.current_index
    
    print("Patch instructions:")
    print("1. Add state tracking to prevent multiple trades at same timestamp")
    print("2. Enforce minimum gap between trades (5 time steps)")
    print("3. Track traded states to prevent exploitation")
    print("4. Add per-episode trade limits based on episode length")
    
    return True

if __name__ == "__main__":
    apply_infinite_trading_patch()
'''
            
            with open('patch_infinite_trading.py', 'w') as f:
                f.write(patch_content)
            
            logger.info("Patch file created: patch_infinite_trading.py")
            
            # Create a monitoring script
            monitor_content = '''#!/usr/bin/env python3
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
'''
            
            with open('monitor_infinite_trading.py', 'w') as f:
                f.write(monitor_content)
            
            logger.info("Monitor script created: monitor_infinite_trading.py")
            
            logger.info("\nSummary:")
            logger.info("1. Stopped sessions with excessive trades")
            logger.info("2. Created patch instructions")
            logger.info("3. Created monitoring script")
            logger.info("\nTo prevent this issue:")
            logger.info("- Apply the patch to RealisticFuturesEnv")
            logger.info("- Run the monitor script to detect exploitation")
            logger.info("- Implement state-based trade limiting")
            
        except Exception as e:
            logger.error(f"Error fixing infinite trading: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    fix_infinite_trading()