import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class RiskLimits:
    max_position_size: int = 5
    max_daily_loss: float = 1000.0
    max_drawdown: float = 0.15  # 15%
    max_leverage: float = 10.0
    max_correlation: float = 0.7
    var_limit: float = 0.02  # 2% VaR
    max_concentration: float = 0.25  # 25% of portfolio

@dataclass
class PositionRisk:
    symbol: str
    position_size: int
    entry_price: float
    current_price: float
    unrealized_pnl: float
    risk_amount: float
    var_1d: float
    var_5d: float
    beta: float
    correlation: float

class RiskManager:
    """
    Advanced risk management system with multiple risk models
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        self.risk_limits = risk_limits or RiskLimits()
        self.positions = {}
        self.daily_pnl = 0.0
        self.portfolio_value = 100000.0  # Initial portfolio value
        self.max_portfolio_value = self.portfolio_value
        self.risk_metrics = {}
        self.risk_alerts = []
        
        # Market data for risk calculations
        self.historical_returns = {}
        self.correlation_matrix = None
        self.volatility_models = {}
        
        logger.info("Risk Manager initialized")
    
    def update_position(self, symbol: str, position_size: int, entry_price: float, 
                       current_price: float) -> bool:
        """
        Update position information
        
        Args:
            symbol: Symbol name
            position_size: Current position size (positive for long, negative for short)
            entry_price: Entry price
            current_price: Current market price
            
        Returns:
            True if position is within risk limits
        """
        try:
            # Calculate unrealized PnL
            if position_size > 0:  # Long position
                unrealized_pnl = (current_price - entry_price) * position_size * 5.0  # NQ tick value
            else:  # Short position
                unrealized_pnl = (entry_price - current_price) * abs(position_size) * 5.0
            
            # Calculate risk metrics
            risk_amount = abs(position_size) * current_price * 5.0 * 0.01  # 1% risk per contract
            var_1d = self.calculate_var(symbol, abs(position_size), current_price, 1)
            var_5d = self.calculate_var(symbol, abs(position_size), current_price, 5)
            
            # Update position
            self.positions[symbol] = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                entry_price=entry_price,
                current_price=current_price,
                unrealized_pnl=unrealized_pnl,
                risk_amount=risk_amount,
                var_1d=var_1d,
                var_5d=var_5d,
                beta=self.calculate_beta(symbol),
                correlation=self.calculate_correlation(symbol)
            )
            
            # Check risk limits
            risk_check = self.check_risk_limits(symbol)
            
            if not risk_check:
                self.add_risk_alert(f"Position {symbol} exceeds risk limits", RiskLevel.HIGH)
            
            return risk_check
            
        except Exception as e:
            logger.error(f"Error updating position: {str(e)}")
            return False
    
    def check_trade_risk(self, symbol: str, proposed_size: int, price: float) -> Tuple[bool, str]:
        """
        Check if a proposed trade meets risk criteria
        
        Args:
            symbol: Symbol to trade
            proposed_size: Proposed position size
            price: Trade price
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            # Check position size limit
            current_position = self.positions.get(symbol, PositionRisk(symbol, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            new_position_size = current_position.position_size + proposed_size
            
            if abs(new_position_size) > self.risk_limits.max_position_size:
                return False, f"Position size {abs(new_position_size)} exceeds limit {self.risk_limits.max_position_size}"
            
            # Check daily loss limit
            proposed_risk = abs(proposed_size) * price * 5.0 * 0.02  # 2% risk per contract
            if self.daily_pnl - proposed_risk < -self.risk_limits.max_daily_loss:
                return False, f"Trade would exceed daily loss limit"
            
            # Check portfolio concentration
            total_exposure = sum(abs(pos.position_size) * pos.current_price * 5.0 
                               for pos in self.positions.values())
            proposed_exposure = abs(proposed_size) * price * 5.0
            
            if (total_exposure + proposed_exposure) / self.portfolio_value > self.risk_limits.max_concentration:
                return False, f"Trade would exceed concentration limit"
            
            # Check VaR limit
            proposed_var = self.calculate_var(symbol, abs(proposed_size), price, 1)
            total_var = sum(pos.var_1d for pos in self.positions.values()) + proposed_var
            
            if total_var / self.portfolio_value > self.risk_limits.var_limit:
                return False, f"Trade would exceed VaR limit"
            
            # Check correlation limits
            if len(self.positions) > 0:
                avg_correlation = np.mean([pos.correlation for pos in self.positions.values()])
                if avg_correlation > self.risk_limits.max_correlation:
                    return False, f"Portfolio correlation {avg_correlation:.2f} exceeds limit {self.risk_limits.max_correlation}"
            
            return True, "Trade approved"
            
        except Exception as e:
            logger.error(f"Error checking trade risk: {str(e)}")
            return False, f"Risk check error: {str(e)}"
    
    def calculate_var(self, symbol: str, position_size: int, price: float, days: int) -> float:
        """
        Calculate Value at Risk using historical simulation
        
        Args:
            symbol: Symbol name
            position_size: Position size
            price: Current price
            days: Number of days
            
        Returns:
            VaR value
        """
        try:
            if symbol not in self.historical_returns:
                # Use default volatility if no historical data
                daily_vol = 0.02  # 2% daily volatility assumption
                return position_size * price * 5.0 * daily_vol * np.sqrt(days) * 1.96  # 95% confidence
            
            returns = self.historical_returns[symbol]
            
            # Calculate portfolio value changes
            portfolio_changes = []
            for i in range(len(returns) - days + 1):
                period_returns = returns[i:i+days]
                period_return = np.prod(1 + period_returns) - 1
                portfolio_change = position_size * price * 5.0 * period_return
                portfolio_changes.append(portfolio_change)
            
            # Calculate VaR at 95% confidence level
            var_95 = np.percentile(portfolio_changes, 5)
            
            return abs(var_95)
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {str(e)}")
            return 0.0
    
    def calculate_beta(self, symbol: str) -> float:
        """Calculate beta relative to market"""
        try:
            if symbol not in self.historical_returns:
                return 1.0  # Default beta
            
            # For futures, beta is typically close to 1.0 relative to the underlying index
            # This is a simplified calculation
            return 1.0
            
        except Exception as e:
            logger.error(f"Error calculating beta: {str(e)}")
            return 1.0
    
    def calculate_correlation(self, symbol: str) -> float:
        """Calculate correlation with existing positions"""
        try:
            if len(self.positions) == 0:
                return 0.0
            
            # Simplified correlation calculation
            # In practice, you would calculate correlation matrix between all positions
            return 0.5  # Default moderate correlation
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
    
    def check_risk_limits(self, symbol: str) -> bool:
        """Check if position is within risk limits"""
        try:
            if symbol not in self.positions:
                return True
            
            position = self.positions[symbol]
            
            # Check position size
            if abs(position.position_size) > self.risk_limits.max_position_size:
                return False
            
            # Check daily loss
            if self.daily_pnl < -self.risk_limits.max_daily_loss:
                return False
            
            # Check drawdown
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.risk_limits.max_drawdown:
                return False
            
            # Check VaR
            if position.var_1d / self.portfolio_value > self.risk_limits.var_limit:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return False
    
    def calculate_position_size(self, symbol: str, price: float, risk_per_trade: float) -> int:
        """
        Calculate optimal position size based on risk management
        
        Args:
            symbol: Symbol to trade
            price: Current price
            risk_per_trade: Risk amount per trade
            
        Returns:
            Recommended position size
        """
        try:
            # Kelly Criterion implementation
            win_rate = 0.55  # Assumed win rate
            avg_win = risk_per_trade * 1.5  # Risk-reward ratio
            avg_loss = risk_per_trade
            
            # Kelly fraction
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Conservative scaling (use 25% of Kelly)
            conservative_fraction = kelly_fraction * 0.25
            
            # Calculate position size
            account_risk = self.portfolio_value * conservative_fraction
            position_size = int(account_risk / (price * 5.0))  # NQ tick value
            
            # Apply maximum position size limit
            max_size = self.risk_limits.max_position_size
            position_size = min(position_size, max_size)
            
            return max(1, position_size)  # At least 1 contract
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 1
    
    def update_portfolio_value(self, new_value: float):
        """Update portfolio value and track high water mark"""
        try:
            self.portfolio_value = new_value
            self.max_portfolio_value = max(self.max_portfolio_value, new_value)
            
            # Update risk metrics
            self.update_risk_metrics()
            
        except Exception as e:
            logger.error(f"Error updating portfolio value: {str(e)}")
    
    def update_daily_pnl(self, pnl: float):
        """Update daily PnL"""
        try:
            self.daily_pnl = pnl
            
            # Check daily loss limit
            if self.daily_pnl < -self.risk_limits.max_daily_loss:
                self.add_risk_alert("Daily loss limit exceeded", RiskLevel.EXTREME)
            
        except Exception as e:
            logger.error(f"Error updating daily PnL: {str(e)}")
    
    def update_risk_metrics(self):
        """Update comprehensive risk metrics"""
        try:
            total_exposure = sum(abs(pos.position_size) * pos.current_price * 5.0 
                               for pos in self.positions.values())
            
            total_var = sum(pos.var_1d for pos in self.positions.values())
            
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            
            leverage = total_exposure / self.portfolio_value if self.portfolio_value > 0 else 0
            
            self.risk_metrics = {
                'total_exposure': total_exposure,
                'total_var': total_var,
                'var_percentage': (total_var / self.portfolio_value) * 100 if self.portfolio_value > 0 else 0,
                'current_drawdown': current_drawdown * 100,
                'leverage': leverage,
                'daily_pnl': self.daily_pnl,
                'portfolio_value': self.portfolio_value,
                'max_portfolio_value': self.max_portfolio_value,
                'risk_level': self.get_overall_risk_level()
            }
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {str(e)}")
    
    def get_overall_risk_level(self) -> RiskLevel:
        """Determine overall portfolio risk level"""
        try:
            risk_score = 0
            
            # Check various risk factors
            if self.daily_pnl < -self.risk_limits.max_daily_loss * 0.5:
                risk_score += 1
            if self.daily_pnl < -self.risk_limits.max_daily_loss * 0.8:
                risk_score += 2
            
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.risk_limits.max_drawdown * 0.5:
                risk_score += 1
            if current_drawdown > self.risk_limits.max_drawdown * 0.8:
                risk_score += 2
            
            total_var = sum(pos.var_1d for pos in self.positions.values())
            var_percentage = (total_var / self.portfolio_value) if self.portfolio_value > 0 else 0
            if var_percentage > self.risk_limits.var_limit * 0.5:
                risk_score += 1
            if var_percentage > self.risk_limits.var_limit * 0.8:
                risk_score += 2
            
            # Determine risk level
            if risk_score >= 4:
                return RiskLevel.EXTREME
            elif risk_score >= 3:
                return RiskLevel.HIGH
            elif risk_score >= 2:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
                
        except Exception as e:
            logger.error(f"Error determining risk level: {str(e)}")
            return RiskLevel.MEDIUM
    
    def add_risk_alert(self, message: str, level: RiskLevel):
        """Add a risk alert"""
        try:
            alert = {
                'timestamp': datetime.now(),
                'message': message,
                'level': level,
                'acknowledged': False
            }
            
            self.risk_alerts.append(alert)
            
            # Log based on severity
            if level == RiskLevel.EXTREME:
                logger.critical(f"EXTREME RISK ALERT: {message}")
            elif level == RiskLevel.HIGH:
                logger.error(f"HIGH RISK ALERT: {message}")
            elif level == RiskLevel.MEDIUM:
                logger.warning(f"MEDIUM RISK ALERT: {message}")
            else:
                logger.info(f"LOW RISK ALERT: {message}")
            
        except Exception as e:
            logger.error(f"Error adding risk alert: {str(e)}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        try:
            return {
                'risk_metrics': self.risk_metrics,
                'risk_limits': {
                    'max_position_size': self.risk_limits.max_position_size,
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'max_drawdown': self.risk_limits.max_drawdown * 100,
                    'var_limit': self.risk_limits.var_limit * 100
                },
                'positions': {symbol: {
                    'position_size': pos.position_size,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'var_1d': pos.var_1d,
                    'risk_amount': pos.risk_amount
                } for symbol, pos in self.positions.items()},
                'alerts': [{
                    'timestamp': alert['timestamp'].isoformat(),
                    'message': alert['message'],
                    'level': alert['level'].value,
                    'acknowledged': alert['acknowledged']
                } for alert in self.risk_alerts[-10:]],  # Last 10 alerts
                'recommendations': self.get_risk_recommendations()
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {str(e)}")
            return {}
    
    def get_risk_recommendations(self) -> List[str]:
        """Get risk management recommendations"""
        try:
            recommendations = []
            
            # Check daily loss
            if self.daily_pnl < -self.risk_limits.max_daily_loss * 0.7:
                recommendations.append("Consider reducing position sizes due to daily loss approaching limit")
            
            # Check drawdown
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.risk_limits.max_drawdown * 0.7:
                recommendations.append("Portfolio drawdown is high - consider defensive positioning")
            
            # Check VaR
            total_var = sum(pos.var_1d for pos in self.positions.values())
            var_percentage = (total_var / self.portfolio_value) if self.portfolio_value > 0 else 0
            if var_percentage > self.risk_limits.var_limit * 0.7:
                recommendations.append("Portfolio VaR is elevated - consider hedging positions")
            
            # Check concentration
            if len(self.positions) == 1:
                recommendations.append("Consider diversifying across multiple positions")
            
            if not recommendations:
                recommendations.append("Risk profile is within acceptable limits")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return ["Error generating recommendations"]
    
    def should_stop_trading(self) -> bool:
        """Determine if trading should be stopped due to risk"""
        try:
            # Stop trading if daily loss limit exceeded
            if self.daily_pnl < -self.risk_limits.max_daily_loss:
                return True
            
            # Stop trading if drawdown limit exceeded
            current_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if current_drawdown > self.risk_limits.max_drawdown:
                return True
            
            # Stop trading if overall risk level is extreme
            if self.get_overall_risk_level() == RiskLevel.EXTREME:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking stop trading condition: {str(e)}")
            return True  # Conservative approach - stop on error
