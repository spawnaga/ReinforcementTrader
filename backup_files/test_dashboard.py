"""
Test script to create a trading session with sample trades to test the dashboard
"""
import os
import sys
from datetime import datetime, timedelta
import random
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import TradingSession, Trade, MarketData
from extensions import db

# Database connection
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

def create_test_session():
    """Create a test trading session with sample trades"""
    db_session = SessionLocal()
    
    try:
        # Create a new trading session
        session = TradingSession(
            session_name="Test Trading Session - Dashboard Demo",
            algorithm_type="ANE-PPO",
            status="active",
            parameters={
                "learning_rate": 0.0003,
                "batch_size": 64,
                "episodes": 1000,
                "risk_level": "moderate"
            },
            total_episodes=1000,
            current_episode=500
        )
        db_session.add(session)
        db_session.commit()
        
        print(f"Created trading session: {session.id}")
        
        # Create sample trades
        base_price = 15000
        trades = []
        
        for i in range(20):
            entry_time = datetime.utcnow() - timedelta(hours=20-i)
            exit_time = entry_time + timedelta(minutes=random.randint(5, 60))
            
            # Simulate price movement
            entry_price = base_price + random.uniform(-100, 100)
            price_change = random.uniform(-50, 50)
            exit_price = entry_price + price_change
            
            # Determine position type
            position_type = "long" if random.random() > 0.5 else "short"
            
            # Calculate profit/loss
            if position_type == "long":
                profit_loss = (exit_price - entry_price) * 5  # NQ point value
            else:
                profit_loss = (entry_price - exit_price) * 5
            
            trade = Trade(
                session_id=session.id,
                trade_id=f"TRADE_{i+1:04d}",
                position_type=position_type,
                quantity=random.randint(1, 5),
                entry_price=entry_price,
                exit_price=exit_price,
                entry_time=entry_time,
                exit_time=exit_time,
                profit_loss=profit_loss,
                status="closed",
                episode_number=random.randint(100, 500)
            )
            trades.append(trade)
        
        # Add all trades
        db_session.bulk_save_objects(trades)
        
        # Update session metrics
        total_profit = sum(t.profit_loss for t in trades)
        win_rate = len([t for t in trades if t.profit_loss > 0]) / len(trades)
        
        session.total_profit = total_profit
        session.total_trades = len(trades)
        session.win_rate = win_rate
        session.current_episode = 500
        session.total_episodes = 1000
        session.sharpe_ratio = 1.5
        session.max_drawdown = 0.08
        
        db_session.commit()
        
        print(f"Created {len(trades)} sample trades")
        print(f"Total profit: ${total_profit:.2f}")
        print(f"Win rate: {win_rate*100:.1f}%")
        
        return session.id
        
    except Exception as e:
        print(f"Error creating test session: {e}")
        db_session.rollback()
        raise
    finally:
        db_session.close()

if __name__ == "__main__":
    session_id = create_test_session()
    print(f"\nTest session created successfully!")
    print(f"Session ID: {session_id}")
    print(f"\nYou can now open the trading dashboard and select this session")