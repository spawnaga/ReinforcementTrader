"""
Script to populate the database with sample trading data
"""
from app import app
from extensions import db
from models import TradingSession, Trade, MarketData, TrainingMetrics
from datetime import datetime, timedelta
import random

def create_sample_data():
    with app.app_context():
        # Create a sample trading session
        session = TradingSession(
            session_name="Sample Trading Session",
            algorithm_type="ANE_PPO",
            status="completed",
            parameters={"learning_rate": 0.0003, "gamma": 0.99},
            start_time=datetime.utcnow() - timedelta(hours=2),
            end_time=datetime.utcnow() - timedelta(hours=1),
            total_episodes=100,
            current_episode=100,
            total_profit=1250.50,
            total_trades=25,
            win_rate=0.68,
            sharpe_ratio=1.45,
            max_drawdown=0.15
        )
        db.session.add(session)
        db.session.commit()
        
        # Create sample trades
        base_price = 23000
        for i in range(10):
            entry_time = datetime.utcnow() - timedelta(minutes=30-i*3)
            exit_time = entry_time + timedelta(minutes=2)
            
            entry_price = base_price + random.uniform(-50, 50)
            exit_price = entry_price + random.uniform(-10, 15)
            
            trade_type = random.choice(['long', 'short'])
            if trade_type == 'short':
                profit = (entry_price - exit_price) * 5 - 5  # $5 per tick minus commission
            else:
                profit = (exit_price - entry_price) * 5 - 5
            
            trade = Trade(
                session_id=session.id,
                trade_id=f"TRADE_{session.id}_{i+1}",
                position_type=trade_type,
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=exit_time,
                exit_price=exit_price,
                quantity=1,
                profit_loss=profit,
                status='closed',
                episode_number=i+1
            )
            db.session.add(trade)
        
        db.session.commit()
        print("Sample data created successfully!")
        
        # Verify data
        trades = Trade.query.all()
        print(f"Total trades in database: {len(trades)}")

if __name__ == "__main__":
    create_sample_data()