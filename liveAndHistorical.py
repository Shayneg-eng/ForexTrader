from datetime import datetime, timedelta
import MetaTrader5 as mt5
import logging
import numpy as np
from collections import namedtuple
import time
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a data structure for OHLCV data
Bar = namedtuple('Bar', ['time', 'open', 'high', 'low', 'close', 'tick_volume'])

class TradingConfig:
    SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5
    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE = 0.1

    EMA_FAST = 5
    EMA_MEDIUM = 12
    EMA_SLOW = 26
    RSI_PERIOD = 8
    RSI_OVERBOUGHT = 65
    RSI_OVERSOLD = 35
    MACD_FAST = 8
    MACD_SLOW = 17
    MACD_SIGNAL = 6
    ATR_PERIOD = 10
    STOP_LOSS_ATR_MULT = 0.75
    TAKE_PROFIT_ATR_MULT = 1.0
    MAX_RISK_PER_TRADE = 0.005
    MIN_VOLUME_RATIO = 0.8
    MAX_SPREAD = 20
    DEVIATION = 20
    
class MT5Handler:
    def initialize(self) -> bool:
        if not mt5.initialize():
            logger.error("MT5 initialization failed")
            return False
        logger.info("MT5 initialized successfully")
        return True

    def get_historical_bars(self, symbol: str, timeframe: int, num_bars: int) -> list:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        return [Bar(datetime.fromtimestamp(rate[0]), rate[1], rate[2], rate[3], rate[4], rate[5]) for rate in rates]

    def get_historical_range(self, symbol: str, timeframe: int, days_back: int) -> list:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        return [Bar(datetime.fromtimestamp(rate[0]), rate[1], rate[2], rate[3], rate[4], rate[5]) for rate in rates]

    def get_current_price(self, symbol: str) -> tuple:
        tick = mt5.symbol_info_tick(symbol)
        return tick.bid, tick.ask

    def place_order(self, order_type: int, symbol: str, volume: float, price: float, sl: float, tp: float):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": TradingConfig.DEVIATION,
            "magic": 234000,
            "comment": "live trader",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment}")
            return False
        return True
    
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
    def calculate_indicators(bars: list) -> dict:
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

            # MACD
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
            
            # Buy conditions
            buy_conditions = (
                (indicators['EMA_fast'][latest_idx] > indicators['EMA_medium'][latest_idx]) and
                (indicators['EMA_medium'][latest_idx] > indicators['EMA_slow'][latest_idx]) and
                (indicators['RSI'][latest_idx] > TradingConfig.RSI_OVERSOLD) and
                (indicators['RSI'][latest_idx] < 60) and
                (indicators['MACD_hist'][latest_idx] > 0) and
                (indicators['Volume_ratio'][latest_idx] > TradingConfig.MIN_VOLUME_RATIO)
            )

            # Sell conditions
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
        
class StrategyExecutor:
    def __init__(self, mode: str, mt5_handler: MT5Handler):
        self.mode = mode
        self.mt5_handler = mt5_handler
        self.bars = []
        self.balance = TradingConfig.INITIAL_CAPITAL

    def initialize(self):
        if not self.mt5_handler.initialize():
            return False

        if self.mode == "live":
            # Get recent bars for initialization
            self.bars = self.mt5_handler.get_historical_bars(TradingConfig.SYMBOL, TradingConfig.TIMEFRAME, 100)
        elif self.mode == "historical":
            # Get historical data for backtesting
            self.bars = self.mt5_handler.get_historical_range(TradingConfig.SYMBOL, TradingConfig.TIMEFRAME, 100)
        
        return True

    def execute(self):
        if self.mode == "live":
            self.run_live_trading()
        elif self.mode == "historical":
            self.run_backtest()

    def run_live_trading(self):
        while True:
            indicators = TechnicalIndicators.calculate_indicators(self.bars)
            signal = SignalGenerator.generate_signal(indicators)
            if signal != 0:
                self.execute_trade(signal, indicators)
            # Fetch new bar every minute
            time.sleep(60)

    def run_backtest(self):
        indicators = TechnicalIndicators.calculate_indicators(self.bars)
        signals = [SignalGenerator.generate_signal(indicators) for _ in range(len(self.bars))]
        self.simulate_trades(indicators, signals)

    def execute_trade(self, signal, indicators):
        bid, ask = self.mt5_handler.get_current_price(TradingConfig.SYMBOL)
        current_price = (bid + ask) / 2

        stop_loss_distance = indicators['ATR'][-1] * TradingConfig.STOP_LOSS_ATR_MULT
        tp_distance = indicators['ATR'][-1] * TradingConfig.TAKE_PROFIT_ATR_MULT

        sl = current_price - stop_loss_distance if signal == 1 else current_price + stop_loss_distance
        tp = current_price + tp_distance if signal == 1 else current_price - tp_distance

        if self.mode == "live":
            self.mt5_handler.place_order(mt5.ORDER_TYPE_BUY if signal == 1 else mt5.ORDER_TYPE_SELL, 
                                         TradingConfig.SYMBOL, TradingConfig.POSITION_SIZE, current_price, sl, tp)
        else:
            # Simulate trade execution in backtest
            pass

    def simulate_trades(self, indicators, signals):
        capital = TradingConfig.INITIAL_CAPITAL
        equity_curve = [capital]
        # Simulate the trades here based on signals
        # Log the results
        logger.info("Backtest completed.")
        
def main():
    mode = input("Enter 'live' for live trading or 'historical' for backtesting: ").strip().lower()
    if mode not in ["live", "historical"]:
        logger.error("Invalid mode selected. Please choose 'live' or 'historical'.")
        return

    mt5_handler = MT5Handler()
    executor = StrategyExecutor(mode, mt5_handler)

    if executor.initialize():
        executor.execute()
    else:
        logger.error("Failed to initialize trading strategy.")

if __name__ == "__main__":
    main()