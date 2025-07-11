import asyncio
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from ib_insync import IB, Stock, Future, MarketOrder, LimitOrder, Contract, util
from ib_insync.objects import BarData, TickData
import threading
import time

logger = logging.getLogger(__name__)

class IBIntegration:
    """
    Interactive Brokers integration for live trading and data
    """
    
    def __init__(self):
        self.ib = IB()
        self.connected = False
        self.contracts = {}
        self.market_data_callbacks = []
        self.order_callbacks = []
        self.connection_params = {}
        
        # NQ futures contract specification
        self.nq_contract = Future(
            symbol='NQ',
            lastTradeDateOrContractMonth='',
            exchange='GLOBEX',
            currency='USD'
        )
        
        # Data storage
        self.live_bars = {}
        self.live_ticks = {}
        self.positions = {}
        self.orders = {}
        
        # Event handlers
        self.ib.errorEvent += self._on_error
        self.ib.orderStatusEvent += self._on_order_status
        self.ib.execDetailsEvent += self._on_execution
        self.ib.positionEvent += self._on_position
        
        logger.info("IB Integration initialized")
    
    def connect(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1) -> bool:
        """
        Connect to Interactive Brokers TWS or IB Gateway
        
        Args:
            host: IB TWS/Gateway host
            port: IB TWS/Gateway port (7497 for TWS, 4002 for Gateway)
            client_id: Unique client ID
            
        Returns:
            Connection success status
        """
        try:
            if self.connected:
                logger.warning("Already connected to IB")
                return True
            
            self.connection_params = {
                'host': host,
                'port': port,
                'client_id': client_id
            }
            
            # Connect to IB
            self.ib.connect(host, port, client_id)
            
            # Wait for connection
            self.ib.sleep(1)
            
            if self.ib.isConnected():
                self.connected = True
                logger.info(f"Connected to IB at {host}:{port} with client ID {client_id}")
                
                # Initialize contracts
                self._initialize_contracts()
                
                # Start data collection
                self._start_data_collection()
                
                return True
            else:
                logger.error("Failed to connect to IB")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to IB: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from Interactive Brokers"""
        try:
            if self.connected:
                self.ib.disconnect()
                self.connected = False
                logger.info("Disconnected from IB")
        except Exception as e:
            logger.error(f"Error disconnecting from IB: {str(e)}")
    
    def _initialize_contracts(self):
        """Initialize and qualify contracts"""
        try:
            # Qualify NQ contract
            self.ib.qualifyContracts(self.nq_contract)
            self.contracts['NQ'] = self.nq_contract
            
            logger.info(f"Initialized contracts: {list(self.contracts.keys())}")
            
        except Exception as e:
            logger.error(f"Error initializing contracts: {str(e)}")
    
    def _start_data_collection(self):
        """Start collecting live market data"""
        try:
            if 'NQ' in self.contracts:
                # Request live bars (1 minute)
                bars = self.ib.reqRealTimeBars(
                    self.contracts['NQ'],
                    barSize=5,  # 5 seconds
                    whatToShow='TRADES',
                    useRTH=False
                )
                
                # Request tick data
                ticks = self.ib.reqMktData(
                    self.contracts['NQ'],
                    genericTickList='',
                    snapshot=False,
                    regulatorySnapshot=False
                )
                
                # Store references
                self.live_bars['NQ'] = bars
                self.live_ticks['NQ'] = ticks
                
                # Set up callbacks
                bars.updateEvent += self._on_bar_update
                ticks.updateEvent += self._on_tick_update
                
                logger.info("Started live data collection for NQ")
                
        except Exception as e:
            logger.error(f"Error starting data collection: {str(e)}")
    
    def _on_bar_update(self, bars, hasNewBar):
        """Handle real-time bar updates"""
        try:
            if hasNewBar:
                bar = bars[-1]
                
                # Convert to standard format
                bar_data = {
                    'timestamp': bar.time,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'symbol': 'NQ'
                }
                
                # Notify callbacks
                for callback in self.market_data_callbacks:
                    try:
                        callback(bar_data)
                    except Exception as e:
                        logger.error(f"Error in market data callback: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error handling bar update: {str(e)}")
    
    def _on_tick_update(self, ticks):
        """Handle tick data updates"""
        try:
            # Process tick data
            tick_data = {
                'timestamp': datetime.now(),
                'bid': ticks.bid,
                'ask': ticks.ask,
                'last': ticks.last,
                'volume': ticks.volume,
                'symbol': 'NQ'
            }
            
            # Store latest tick
            self.live_ticks['NQ_latest'] = tick_data
            
        except Exception as e:
            logger.error(f"Error handling tick update: {str(e)}")
    
    def _on_error(self, reqId, errorCode, errorString, contract):
        """Handle IB errors"""
        logger.error(f"IB Error {errorCode}: {errorString} (ReqId: {reqId})")
    
    def _on_order_status(self, trade):
        """Handle order status updates"""
        try:
            order_data = {
                'order_id': trade.order.orderId,
                'symbol': trade.contract.symbol,
                'action': trade.order.action,
                'quantity': trade.order.totalQuantity,
                'order_type': trade.order.orderType,
                'status': trade.orderStatus.status,
                'filled': trade.orderStatus.filled,
                'remaining': trade.orderStatus.remaining,
                'avg_fill_price': trade.orderStatus.avgFillPrice,
                'timestamp': datetime.now()
            }
            
            # Store order status
            self.orders[trade.order.orderId] = order_data
            
            # Notify callbacks
            for callback in self.order_callbacks:
                try:
                    callback(order_data)
                except Exception as e:
                    logger.error(f"Error in order callback: {str(e)}")
                    
            logger.info(f"Order {trade.order.orderId} status: {trade.orderStatus.status}")
            
        except Exception as e:
            logger.error(f"Error handling order status: {str(e)}")
    
    def _on_execution(self, trade, fill):
        """Handle execution reports"""
        try:
            execution_data = {
                'order_id': trade.order.orderId,
                'symbol': fill.contract.symbol,
                'side': fill.execution.side,
                'quantity': fill.execution.shares,
                'price': fill.execution.price,
                'time': fill.execution.time,
                'exchange': fill.execution.exchange,
                'commission': fill.commissionReport.commission if fill.commissionReport else 0
            }
            
            logger.info(f"Execution: {execution_data}")
            
        except Exception as e:
            logger.error(f"Error handling execution: {str(e)}")
    
    def _on_position(self, position):
        """Handle position updates"""
        try:
            position_data = {
                'symbol': position.contract.symbol,
                'position': position.position,
                'avg_cost': position.avgCost,
                'market_price': position.marketPrice,
                'market_value': position.marketValue,
                'unrealized_pnl': position.unrealizedPNL,
                'realized_pnl': position.realizedPNL
            }
            
            self.positions[position.contract.symbol] = position_data
            
            logger.info(f"Position update: {position_data}")
            
        except Exception as e:
            logger.error(f"Error handling position: {str(e)}")
    
    def place_market_order(self, symbol: str, action: str, quantity: int) -> Optional[int]:
        """
        Place a market order
        
        Args:
            symbol: Symbol to trade
            action: 'BUY' or 'SELL'
            quantity: Number of contracts
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if not self.connected:
                logger.error("Not connected to IB")
                return None
            
            if symbol not in self.contracts:
                logger.error(f"Contract {symbol} not found")
                return None
            
            # Create market order
            order = MarketOrder(action, quantity)
            
            # Place order
            trade = self.ib.placeOrder(self.contracts[symbol], order)
            
            order_id = trade.order.orderId
            
            logger.info(f"Placed market order: {action} {quantity} {symbol}, Order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing market order: {str(e)}")
            return None
    
    def place_limit_order(self, symbol: str, action: str, quantity: int, price: float) -> Optional[int]:
        """
        Place a limit order
        
        Args:
            symbol: Symbol to trade
            action: 'BUY' or 'SELL'
            quantity: Number of contracts
            price: Limit price
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if not self.connected:
                logger.error("Not connected to IB")
                return None
            
            if symbol not in self.contracts:
                logger.error(f"Contract {symbol} not found")
                return None
            
            # Create limit order
            order = LimitOrder(action, quantity, price)
            
            # Place order
            trade = self.ib.placeOrder(self.contracts[symbol], order)
            
            order_id = trade.order.orderId
            
            logger.info(f"Placed limit order: {action} {quantity} {symbol} @ {price}, Order ID: {order_id}")
            
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing limit order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order"""
        try:
            if not self.connected:
                logger.error("Not connected to IB")
                return False
            
            # Find the trade
            trade = None
            for t in self.ib.trades():
                if t.order.orderId == order_id:
                    trade = t
                    break
            
            if not trade:
                logger.error(f"Order {order_id} not found")
                return False
            
            # Cancel the order
            self.ib.cancelOrder(trade.order)
            
            logger.info(f"Cancelled order {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            if not self.connected:
                return {}
            
            positions = {}
            for position in self.ib.positions():
                positions[position.contract.symbol] = {
                    'position': position.position,
                    'avg_cost': position.avgCost,
                    'market_price': position.marketPrice,
                    'market_value': position.marketValue,
                    'unrealized_pnl': position.unrealizedPNL,
                    'realized_pnl': position.realizedPNL
                }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    def get_account_summary(self) -> Dict:
        """Get account summary"""
        try:
            if not self.connected:
                return {}
            
            account_values = self.ib.accountSummary()
            
            summary = {}
            for av in account_values:
                summary[av.tag] = {
                    'value': av.value,
                    'currency': av.currency
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting account summary: {str(e)}")
            return {}
    
    def get_historical_data(self, symbol: str, duration: str = '1 D', 
                           bar_size: str = '1 min', what_to_show: str = 'TRADES') -> Optional[pd.DataFrame]:
        """
        Get historical data
        
        Args:
            symbol: Symbol to get data for
            duration: Duration string (e.g., '1 D', '1 W', '1 M')
            bar_size: Bar size (e.g., '1 min', '5 mins', '1 hour')
            what_to_show: Data type ('TRADES', 'MIDPOINT', 'BID', 'ASK')
            
        Returns:
            DataFrame with historical data
        """
        try:
            if not self.connected:
                logger.error("Not connected to IB")
                return None
            
            if symbol not in self.contracts:
                logger.error(f"Contract {symbol} not found")
                return None
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                self.contracts[symbol],
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=False,
                formatDate=1
            )
            
            if not bars:
                logger.warning(f"No historical data received for {symbol}")
                return None
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.date,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                })
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} historical bars for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return None
    
    def add_market_data_callback(self, callback: Callable):
        """Add callback for market data updates"""
        self.market_data_callbacks.append(callback)
    
    def add_order_callback(self, callback: Callable):
        """Add callback for order updates"""
        self.order_callbacks.append(callback)
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if symbol in self.live_ticks and f"{symbol}_latest" in self.live_ticks:
                return self.live_ticks[f"{symbol}_latest"].get('last')
            return None
        except Exception as e:
            logger.error(f"Error getting current price: {str(e)}")
            return None
    
    def is_market_open(self) -> bool:
        """Check if market is open"""
        try:
            # NQ futures trade almost 24/7, but we can check for active sessions
            now = datetime.now()
            
            # Basic check - futures are generally active except for brief maintenance windows
            # You can implement more sophisticated market hours logic here
            
            return True  # NQ futures trade nearly 24/7
            
        except Exception as e:
            logger.error(f"Error checking market status: {str(e)}")
            return False
    
    def get_connection_status(self) -> Dict:
        """Get connection status information"""
        return {
            'connected': self.connected,
            'connection_params': self.connection_params,
            'contracts': list(self.contracts.keys()),
            'live_data_active': len(self.live_bars) > 0,
            'positions': len(self.positions),
            'active_orders': len([o for o in self.orders.values() if o.get('status') in ['Submitted', 'PreSubmitted']])
        }
