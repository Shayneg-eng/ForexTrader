from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import MetaTrader5 as mt5
import logging
from typing import Dict, Tuple, Optional
from collections import namedtuple
import time
from threading import Thread, Lock
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create data structure for OHLCV data
Bar = namedtuple('Bar', ['time', 'open', 'high', 'low', 'close', 'tick_volume'])

class TradingConfig:
    # Trading Settings
    SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5
    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE = 0.1
    
    MIN_LOTS = 0.01  # Minimum lot size
    MAX_LOTS = 1.0   # Maximum lot size
    LOT_STEP = 0.01  # Lot size increment
    
    # Faster-moving EMAs for quicker signals
    EMA_FAST = 5      # Reduced from 8
    EMA_MEDIUM = 12   # Reduced from 21
    EMA_SLOW = 26     # Reduced from 50

    # More sensitive RSI settings
    RSI_PERIOD = 8    # Reduced from 14 for faster response
    RSI_OVERBOUGHT = 65  # Lowered from 70
    RSI_OVERSOLD = 35    # Raised from 30

    # Faster MACD for quicker signals
    MACD_FAST = 8     # Reduced from 12
    MACD_SLOW = 17    # Reduced from 26
    MACD_SIGNAL = 6   # Reduced from 9

    # Shorter ATR for more responsive stops
    ATR_PERIOD = 10   # Reduced from 14

    # Risk Management - Adjusted for smaller, more frequent trades
    STOP_LOSS_ATR_MULT = 0.75   # Tighter stop loss
    TAKE_PROFIT_ATR_MULT = 1.0  # Smaller but faster targets
    MAX_RISK_PER_TRADE = 0.005  # Halved risk per trade (0.5% instead of 1%)
    MIN_VOLUME_RATIO = 0.8      # More lenient volume requirement

    # Live Trading Specific
    WARMUP_BARS = 100  # Number of historical bars to initialize indicators
    UPDATE_INTERVAL = 1  # Seconds between market checks
    MAX_SPREAD = 20  # Maximum allowed spread in points
    DEVIATION = 20  # Maximum price deviation for orders

class MT5Handler:
    def __init__(self):
        self.initialized = False

    def initialize(self) -> bool:
        try:
            if not self.initialized:
                if not mt5.initialize():
                    logger.error("MT5 initialization failed")
                    return False
                logger.info("MT5 initialized successfully")
                self.initialized = True
            return True
        except Exception as e:
            logger.error(f"MT5 initialization error: {str(e)}")
            return False

    def get_historical_bars(self, symbol: str, timeframe: int, num_bars: int) -> Optional[list]:
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
            if rates is None:
                logger.error("Failed to fetch historical data")
                return None
                
            return [Bar(
                datetime.fromtimestamp(rate[0]),
                rate[1], rate[2], rate[3], rate[4], rate[5]
            ) for rate in rates]

        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return None

    def place_order(self, order_type: int, symbol: str, volume: float, price: float, 
                   sl: float, tp: float, deviation: int) -> bool:
        try:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": 234000,
                "comment": "python live trader",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order failed: {result.comment}")
                return False
                
            logger.info(f"Order placed successfully: {order_type}, Volume: {volume}, Price: {price}")
            return True

        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise Exception("Failed to get current price")
        return tick.bid, tick.ask

    def get_position(self, symbol: str) -> Optional[dict]:
        positions = mt5.positions_get(symbol=symbol)
        if positions:
            return {
                "type": positions[0].type,
                "volume": positions[0].volume,
                "price": positions[0].price_open,
                "ticket": positions[0].ticket
            }
        return None

    def close_position(self, ticket: int) -> bool:
        try:
            position = mt5.positions_get(ticket=ticket)
            if not position:
                logger.error("No position found to close")
                return False
            
            # Enable symbol data
            symbol = position[0].symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return False

            # Get fresh price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                logger.error("Failed to get current price")
                return False

            # Determine closing price based on position type
            close_price = tick.bid if position[0].type == mt5.ORDER_TYPE_BUY else tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": position[0].volume,
                "type": mt5.ORDER_TYPE_SELL if position[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": close_price,
                "deviation": TradingConfig.DEVIATION,
                "magic": 234000,
                "comment": "python close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Failed to close position: {result.comment}")
                return False

            logger.info(f"Position closed successfully at {close_price}")
            return True

        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return False
    
class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        for i in range(period, len(prices)):
            avg_gains[i] = np.mean(gains[i-period:i])
            avg_losses[i] = np.mean(losses[i-period:i])
        
        rs = avg_gains / np.where(avg_losses == 0, 1, avg_losses)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
        high_low = high - low
        high_close = np.abs(high - np.roll(close, 1))
        low_close = np.abs(low - np.roll(close, 1))
        ranges = np.maximum(high_low, np.maximum(high_close, low_close))
        return np.array([np.mean(ranges[max(0, i-period):i]) for i in range(1, len(ranges)+1)])

    @staticmethod
    def calculate_all_indicators(bars: list) -> dict:
        try:
            closes = np.array([bar.close for bar in bars])
            highs = np.array([bar.high for bar in bars])
            lows = np.array([bar.low for bar in bars])
            volumes = np.array([bar.tick_volume for bar in bars])

            indicators = {
                'close': closes,
                'EMA_fast': TechnicalIndicators.calculate_ema(closes, TradingConfig.EMA_FAST),
                'EMA_medium': TechnicalIndicators.calculate_ema(closes, TradingConfig.EMA_MEDIUM),
                'EMA_slow': TechnicalIndicators.calculate_ema(closes, TradingConfig.EMA_SLOW),
                'RSI': TechnicalIndicators.calculate_rsi(closes, TradingConfig.RSI_PERIOD),
                'ATR': TechnicalIndicators.calculate_atr(highs, lows, closes, TradingConfig.ATR_PERIOD),
                'Volume_ratio': volumes / np.convolve(volumes, np.ones(20)/20, mode='same')
            }

            # MACD calculation
            ema_fast = TechnicalIndicators.calculate_ema(closes, TradingConfig.MACD_FAST)
            ema_slow = TechnicalIndicators.calculate_ema(closes, TradingConfig.MACD_SLOW)
            indicators['MACD_line'] = ema_fast - ema_slow
            indicators['MACD_signal'] = TechnicalIndicators.calculate_ema(
                indicators['MACD_line'],
                TradingConfig.MACD_SIGNAL
            )
            indicators['MACD_hist'] = indicators['MACD_line'] - indicators['MACD_signal']

            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

class SignalGenerator:
    @staticmethod
    def generate_signal(indicators: dict) -> int:
        try:
            latest_idx = -1
            
            # Buy conditions - exactly matching backtester
            buy_conditions = (
                (indicators['EMA_fast'][latest_idx] > indicators['EMA_medium'][latest_idx]) and
                (indicators['EMA_medium'][latest_idx] > indicators['EMA_slow'][latest_idx]) and
                (indicators['RSI'][latest_idx] > TradingConfig.RSI_OVERSOLD) and
                (indicators['RSI'][latest_idx] < 60) and
                (indicators['MACD_hist'][latest_idx] > 0) and
                (indicators['Volume_ratio'][latest_idx] > TradingConfig.MIN_VOLUME_RATIO)
            )

            # Sell conditions - exactly matching backtester
            sell_conditions = (
                (indicators['EMA_fast'][latest_idx] < indicators['EMA_medium'][latest_idx]) and
                (indicators['EMA_medium'][latest_idx] < indicators['EMA_slow'][latest_idx]) and
                (indicators['RSI'][latest_idx] < TradingConfig.RSI_OVERBOUGHT) and
                (indicators['RSI'][latest_idx] > 40) and
                (indicators['MACD_hist'][latest_idx] < 0) and
                (indicators['Volume_ratio'][latest_idx] > TradingConfig.MIN_VOLUME_RATIO)
            )

            if buy_conditions:
                return 1
            elif sell_conditions:
                return -1
            return 0

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return 0

class PositionSizer:
    @staticmethod
    def calculate_position_size(current_price: float, atr: float, balance: float) -> float:
        """Calculate position size based on ATR and risk parameters"""
        try:
            # Get symbol information
            symbol_info = mt5.symbol_info(TradingConfig.SYMBOL)
            if not symbol_info:
                logger.error("Failed to get symbol info")
                return 0.0

            # Get minimum lot size and lot step
            min_lot = symbol_info.volume_min
            lot_step = symbol_info.volume_step

            # Calculate position size based on risk
            stop_loss_distance = atr * TradingConfig.STOP_LOSS_ATR_MULT
            risk_amount = balance * TradingConfig.MAX_RISK_PER_TRADE
            position_size = risk_amount / stop_loss_distance

            # Limit position size based on config
            position_size = min(position_size, balance * TradingConfig.POSITION_SIZE)

            # Normalize the position size to broker's requirements
            position_size = round(position_size / lot_step) * lot_step
            
            # Ensure position size is at least minimum lot and not more than maximum
            position_size = max(min_lot, min(position_size, symbol_info.volume_max))

            logger.info(f"Calculated position size: {position_size} lots")
            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
    
class LiveTrader:
    def __init__(self):
        self.mt5_handler = MT5Handler()
        self.running = False
        self.bars = []
        self.current_position = None
        self.lock = Lock()
        self.last_bar_time = None
        self.account_info = None
        
        # Performance tracking
        self.daily_trades = []
        self.daily_pnl = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_balance = 0.0
        self.start_of_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Dashboard related
        self.signal_strength = 0
        self.signal_history = []
        self.last_update_time = datetime.now()
        
        # Start the dashboard
        self.start_dashboard()
        
    def manage_positions(self, indicators: dict) -> None:
        try:
            current_position = self.mt5_handler.get_position(TradingConfig.SYMBOL)
            
            if current_position:
                tick = mt5.symbol_info_tick(TradingConfig.SYMBOL)
                if not tick:
                    return

                position_type = current_position['type']
                entry_price = current_position['price']
                current_price = tick.bid if position_type == mt5.ORDER_TYPE_BUY else tick.ask
                
                # Calculate stops exactly like backtester
                stop_loss = entry_price - (position_type * indicators['ATR'][-1] * TradingConfig.STOP_LOSS_ATR_MULT)
                take_profit = entry_price + (position_type * indicators['ATR'][-1] * TradingConfig.TAKE_PROFIT_ATR_MULT)

                if position_type == mt5.ORDER_TYPE_BUY:
                    if current_price <= stop_loss:
                        self.mt5_handler.close_position(current_position['ticket'])
                        self.daily_pnl -= TradingConfig.MAX_RISK_PER_TRADE * self.last_balance
                        self.daily_losses += 1
                    elif current_price >= take_profit:
                        self.mt5_handler.close_position(current_position['ticket'])
                        self.daily_pnl += TradingConfig.MAX_RISK_PER_TRADE * 2 * self.last_balance
                        self.daily_wins += 1
                else:  # SELL position
                    if current_price >= stop_loss:
                        self.mt5_handler.close_position(current_position['ticket'])
                        self.daily_pnl -= TradingConfig.MAX_RISK_PER_TRADE * self.last_balance
                        self.daily_losses += 1
                    elif current_price <= take_profit:
                        self.mt5_handler.close_position(current_position['ticket'])
                        self.daily_pnl += TradingConfig.MAX_RISK_PER_TRADE * 2 * self.last_balance
                        self.daily_wins += 1

        except Exception as e:
            logger.error(f"Error managing positions: {str(e)}")

    def execute_trade(self, signal: int, indicators: dict) -> None:
        try:
            if not self.check_trading_conditions():
                return

            current_price = self.bars[-1].close
            atr = indicators['ATR'][-1]
            
            # Calculate position size exactly like backtester
            stop_loss_distance = atr * TradingConfig.STOP_LOSS_ATR_MULT
            risk_amount = self.account_info.balance * TradingConfig.MAX_RISK_PER_TRADE
            position_size = risk_amount / stop_loss_distance
            position_size = min(position_size, self.account_info.balance * TradingConfig.POSITION_SIZE)

            if signal == 1:  # Buy Signal
                stop_loss = current_price - (atr * TradingConfig.STOP_LOSS_ATR_MULT)
                take_profit = current_price + (atr * TradingConfig.TAKE_PROFIT_ATR_MULT)
                self.mt5_handler.place_order(
                    mt5.ORDER_TYPE_BUY,
                    TradingConfig.SYMBOL,
                    position_size,
                    current_price,
                    stop_loss,
                    take_profit,
                    TradingConfig.DEVIATION
                )
                
            elif signal == -1:  # Sell Signal
                stop_loss = current_price + (atr * TradingConfig.STOP_LOSS_ATR_MULT)
                take_profit = current_price - (atr * TradingConfig.TAKE_PROFIT_ATR_MULT)
                self.mt5_handler.place_order(
                    mt5.ORDER_TYPE_SELL,
                    TradingConfig.SYMBOL,
                    position_size,
                    current_price,
                    stop_loss,
                    take_profit,
                    TradingConfig.DEVIATION
                )

        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")

    def check_trading_conditions(self) -> bool:
        """Check if trading conditions are suitable"""
        try:
            # Check spread
            tick = mt5.symbol_info_tick(TradingConfig.SYMBOL)
            if not tick:
                return False
            
            spread = (tick.ask - tick.bid) * 10000  # Convert to points
            if spread > TradingConfig.MAX_SPREAD:
                logger.warning(f"Spread too high: {spread}")
                return False

            # Check if market is open
            symbol_info = mt5.symbol_info(TradingConfig.SYMBOL)
            if not symbol_info or not symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking trading conditions: {str(e)}")
            return False

    def update_daily_stats(self):
        current_time = datetime.now()
        
        # Reset daily stats at the start of a new day
        if current_time.date() > self.start_of_day.date():
            logger.info(f"Daily Summary - PnL: ${self.daily_pnl:.2f}, Wins: {self.daily_wins}, Losses: {self.daily_losses}")
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0
            self.start_of_day = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            self.last_balance = mt5.account_info().balance

    def trading_loop(self):
        while self.running:
            try:
                # Get and print current price
                bid, ask = self.mt5_handler.get_current_price(TradingConfig.SYMBOL)
                current_balance = mt5.account_info().balance
                
                # Update daily P&L
                daily_change = current_balance - self.last_balance
                if daily_change != self.daily_pnl:
                    if daily_change > self.daily_pnl:
                        self.daily_wins += 1
                    elif daily_change < self.daily_pnl:
                        self.daily_losses += 1
                    self.daily_pnl = daily_change
                
                # Update market data
                new_data = self.update_market_data()
                
                if new_data:
                    indicators = TechnicalIndicators.calculate_all_indicators(self.bars)
                    # Update signal strength
                    self.signal_strength = self.calculate_signal_strength(indicators)
                    self.last_update_time = datetime.now()
                    
                    self.manage_positions(indicators)
                    if not self.mt5_handler.get_position(TradingConfig.SYMBOL):
                        signal = SignalGenerator.generate_signal(indicators)
                        if signal != 0:
                            self.execute_trade(signal, indicators)
                
                # Update daily stats
                self.update_daily_stats()
                
                time.sleep(TradingConfig.UPDATE_INTERVAL)

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(TradingConfig.UPDATE_INTERVAL)

    def update_market_data(self) -> bool:
        try:
            current_bar = self.mt5_handler.get_historical_bars(
                TradingConfig.SYMBOL,
                TradingConfig.TIMEFRAME,
                1
            )
            
            if not current_bar:
                return False

            current_bar = current_bar[0]
            
            with self.lock:
                if current_bar.time > self.last_bar_time:
                    self.bars.append(current_bar)
                    self.bars.pop(0)  # Remove oldest bar
                    self.last_bar_time = current_bar.time
                    return True
            return False

        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
            return False

    def start(self):
        if not self.initialize():
            return False
        
        self.running = True
        Thread(target=self.trading_loop).start()
        logger.info("Live trader started")
        return True

    def stop(self):
        self.running = False
        # Close any open positions
        current_position = self.mt5_handler.get_position(TradingConfig.SYMBOL)
        if current_position:
            self.mt5_handler.close_position(current_position['ticket'])
        mt5.shutdown()
        logger.info("Live trader stopped")

    def initialize(self) -> bool:
        try:
            if not self.mt5_handler.initialize():
                return False

            self.bars = self.mt5_handler.get_historical_bars(
                TradingConfig.SYMBOL,
                TradingConfig.TIMEFRAME,
                TradingConfig.WARMUP_BARS
            )
            
            if not self.bars:
                logger.error("Failed to initialize historical data")
                return False

            self.last_bar_time = self.bars[-1].time
            self.account_info = mt5.account_info()
            if not self.account_info:
                logger.error("Failed to get account info")
                return False

            self.last_balance = self.account_info.balance
            logger.info(f"Initial balance: {self.last_balance}")
            
            logger.info("Live trader initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            return False

    def calculate_signal_strength(self, indicators: dict) -> float:
        """Calculate how close conditions are to generating a trade signal"""
        try:
            latest_idx = -1
            
            # Calculate individual component strengths (-1 to 1 range)
            ema_strength = (
                ((indicators['EMA_fast'][latest_idx] - indicators['EMA_medium'][latest_idx]) +
                (indicators['EMA_medium'][latest_idx] - indicators['EMA_slow'][latest_idx])) / 
                (indicators['ATR'][latest_idx] * 2)
            )
            
            rsi_strength = (indicators['RSI'][latest_idx] - 50) / 50
            
            macd_strength = indicators['MACD_hist'][latest_idx] / indicators['ATR'][latest_idx]
            
            volume_strength = (indicators['Volume_ratio'][latest_idx] - TradingConfig.MIN_VOLUME_RATIO) / 2
            
            # Combine components
            total_strength = (
                ema_strength * 0.4 +
                rsi_strength * 0.3 +
                macd_strength * 0.2 +
                volume_strength * 0.1
            )
            
            # Convert to -100 to 100 range
            return round(total_strength * 100)

        except Exception as e:
            logger.error(f"Error calculating signal strength: {str(e)}")
            return 0

    def calculate_trade_proximity(self, indicators: dict) -> dict:
        """Calculate how close each condition is to triggering a trade"""
        try:
            latest_idx = -1
            
            # Buy conditions proximity
            buy_proximity = {
                'EMA_alignment': (
                    (indicators['EMA_fast'][latest_idx] - indicators['EMA_medium'][latest_idx]) +
                    (indicators['EMA_medium'][latest_idx] - indicators['EMA_slow'][latest_idx])
                ) / (indicators['ATR'][latest_idx] * 2),
                'RSI': (indicators['RSI'][latest_idx] - TradingConfig.RSI_OVERSOLD) / (60 - TradingConfig.RSI_OVERSOLD),
                'MACD': indicators['MACD_hist'][latest_idx] / indicators['ATR'][latest_idx],
                'Volume': (indicators['Volume_ratio'][latest_idx] - TradingConfig.MIN_VOLUME_RATIO) / 1.0
            }
            
            # Sell conditions proximity
            sell_proximity = {
                'EMA_alignment': -(
                    (indicators['EMA_fast'][latest_idx] - indicators['EMA_medium'][latest_idx]) +
                    (indicators['EMA_medium'][latest_idx] - indicators['EMA_slow'][latest_idx])
                ) / (indicators['ATR'][latest_idx] * 2),
                'RSI': (TradingConfig.RSI_OVERBOUGHT - indicators['RSI'][latest_idx]) / (TradingConfig.RSI_OVERBOUGHT - 40),
                'MACD': -indicators['MACD_hist'][latest_idx] / indicators['ATR'][latest_idx],
                'Volume': (indicators['Volume_ratio'][latest_idx] - TradingConfig.MIN_VOLUME_RATIO) / 1.0
            }
            
            return {
                'buy': buy_proximity,
                'sell': sell_proximity
            }
            
        except Exception as e:
            logger.error(f"Error calculating trade proximity: {str(e)}")
            return {'buy': {}, 'sell': {}}
    
    def start_dashboard(self):
            """Create and serve the trading dashboard"""
            class DashboardHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        with open('trading_dashboard.html', 'rb') as file:
                            self.wfile.write(file.read())
                    elif self.path == '/data':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        # Get current price
                        bid, ask = trader.mt5_handler.get_current_price(TradingConfig.SYMBOL)
                        
                        if len(trader.bars) > 0:
                            indicators = TechnicalIndicators.calculate_all_indicators(trader.bars)
                            proximity = trader.calculate_trade_proximity(indicators)
                        else:
                            proximity = {'buy': {}, 'sell': {}}

                        data = {
                            'signal_strength': trader.signal_strength,
                            'current_position': trader.mt5_handler.get_position(TradingConfig.SYMBOL),
                            'daily_pnl': trader.daily_pnl,
                            'daily_wins': trader.daily_wins,
                            'daily_losses': trader.daily_losses,
                            'current_price': {
                                'bid': bid,
                                'ask': ask
                            },
                            'trade_proximity': proximity,
                            'last_update': trader.last_update_time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.wfile.write(json.dumps(data).encode())

            global trader
            trader = self
            
            # Create dashboard HTML file
            dashboard_html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f0f2f5;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
            }
            .gauge { 
                width: 100%; 
                height: 400px; 
                background-color: white;
                border-radius: 10px;
                padding: 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
            }

            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .stats { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                gap: 20px; 
                margin-top: 20px; 
            }
            .stat-box { 
                padding: 20px; 
                background: white; 
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .proximity-bars {
                background: white;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .signal-strong { color: #ff4444; }
            .signal-medium { color: #ffaa00; }
            .signal-weak { color: #00aa00; }
            .price-ticker {
                font-size: 1.5em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .last-update {
                text-align: right;
                color: #666;
                font-size: 0.9em;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Trading Dashboard</h1>
            <div class="gauge" id="gauge"></div>
            <div class="stats">
                <div class="stat-box">
                    <h3>Price</h3>
                    <div id="price" class="price-ticker"></div>
                </div>
                <div class="stat-box">
                    <h3>Daily P&L</h3>
                    <div id="pnl"></div>
                </div>
                <div class="stat-box">
                    <h3>Win Rate</h3>
                    <div id="winrate"></div>
                </div>
                <div class="stat-box">
                    <h3>Position Status</h3>
                    <div id="position"></div>
                </div>
            </div>
            <div class="proximity-bars">
                <h3>Buy Signal Proximity</h3>
                <div id="buyProximity"></div>
                <h3>Sell Signal Proximity</h3>
                <div id="sellProximity"></div>
            </div>
            <div class="last-update" id="lastUpdate"></div>
        </div>

        <script>
            function updateGauge(value) {
                var data = [{
                    type: "indicator",
                    mode: "gauge+number",
                    value: value,
                    title: { 
                        text: "Signal Strength",
                        font: { size: 24 }
                    },
                    gauge: {
                        axis: { 
                            range: [-100, 100],
                            tickwidth: 1,
                            tickcolor: "#666",
                            tickmode: "array",
                            tickvals: [-100, -50, 0, 50, 100],
                            ticktext: ['-100', '-50', '0', '50', '100'],
                            tickangle: 0
                        },
                        shape: "angular",
                        bar: { 
                            color: value > 70 || value < -70 ? "#ff4444" : 
                                value > 50 || value < -50 ? "#ffaa00" : "#00aa00",
                            thickness: 0.3
                        },
                        bgcolor: "white",
                        borderwidth: 2,
                        bordercolor: "#666",
                        steps: [
                            { range: [-100, -70], color: "rgba(255,68,68,0.1)" },
                            { range: [-70, -50], color: "rgba(255,170,0,0.1)" },
                            { range: [-50, 50], color: "rgba(0,170,0,0.1)" },
                            { range: [50, 70], color: "rgba(255,170,0,0.1)" },
                            { range: [70, 100], color: "rgba(255,68,68,0.1)" }
                        ],
                        threshold: {
                            line: { color: "black", width: 4 },
                            thickness: 0.75,
                            value: value
                        }
                    }
                }];

                var layout = { 
                    width: 600,
                    height: 400,
                    margin: { t: 100, b: 0, l: 0, r: 0 },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)",
                    font: { size: 16 }
                };

                var config = {
                    responsive: true,
                    displayModeBar: false
                };

                Plotly.newPlot('gauge', data, layout, config);
            }

            function updateProximityBars(proximity) {
                // Buy Proximity
                var buyData = [{
                    type: 'bar',
                    x: Object.values(proximity.buy),
                    y: Object.keys(proximity.buy),
                    orientation: 'h',
                    marker: {
                        color: Object.values(proximity.buy).map(val => 
                            val > 0.7 ? '#ff4444' : 
                            val > 0.5 ? '#ffaa00' : '#00aa00'
                        )
                    }
                }];

                // Sell Proximity
                var sellData = [{
                    type: 'bar',
                    x: Object.values(proximity.sell),
                    y: Object.keys(proximity.sell),
                    orientation: 'h',
                    marker: {
                        color: Object.values(proximity.sell).map(val => 
                            val > 0.7 ? '#ff4444' : 
                            val > 0.5 ? '#ffaa00' : '#00aa00'
                        )
                    }
                }];

                var layout = {
                    height: 200,
                    margin: { t: 0, b: 30, l: 100, r: 30 },
                    xaxis: { range: [0, 1] },
                    paper_bgcolor: "rgba(0,0,0,0)",
                    plot_bgcolor: "rgba(0,0,0,0)"
                };

                Plotly.newPlot('buyProximity', buyData, layout);
                Plotly.newPlot('sellProximity', sellData, layout);
            }

            function updateStats(data) {
                // Update price
                document.getElementById('price').innerHTML = `
                    Bid: ${data.current_price.bid}<br>
                    Ask: ${data.current_price.ask}
                `;
                
                // Update P&L
                document.getElementById('pnl').textContent = `$${data.daily_pnl.toFixed(2)}`;
                
                // Update win rate
                const winRate = (data.daily_wins / (data.daily_wins + data.daily_losses) * 100) || 0;
                document.getElementById('winrate').textContent = `${winRate.toFixed(1)}%`;
                
                // Update position
                document.getElementById('position').textContent = data.current_position ? 
                    `${data.current_position.type === 0 ? 'BUY' : 'SELL'} ${data.current_position.volume}` : 
                    'No Position';
                    
                // Update last update time
                document.getElementById('lastUpdate').textContent = `Last Update: ${data.last_update}`;
            }

            function updateDashboard() {
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        updateGauge(data.signal_strength);
                        updateStats(data);
                        updateProximityBars(data.trade_proximity);
                    });
            }

            // Update every second
            setInterval(updateDashboard, 1000);
            updateDashboard();
        </script>
    </body>
    </html>
            '''
            
            with open('trading_dashboard.html', 'w') as f:
                f.write(dashboard_html)
            
            # Start dashboard server
            server = HTTPServer(('localhost', 8080), DashboardHandler)
            Thread(target=server.serve_forever, daemon=True).start()
            logger.info("Dashboard started at http://localhost:8080")
            

def main():
    trader = LiveTrader()
    
    try:
        if trader.start():
            logger.info(f"Live trading started for {TradingConfig.SYMBOL}")
            
            # Keep the main thread running
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        trader.stop()

if __name__ == "__main__":
    main()