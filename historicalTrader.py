from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import MetaTrader5 as mt5
import logging
from typing import Dict, Tuple, Optional
from collections import namedtuple
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a data structure for OHLCV data
Bar = namedtuple('Bar', ['time', 'open', 'high', 'low', 'close', 'tick_volume'])

class TradingConfig:
    # Day trading settings
    SYMBOL = "EURUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute chart
    DAYS_TO_ANALYZE = 100  # Analyze recent market behavior
    INITIAL_CAPITAL = 10000.0
    POSITION_SIZE = 0.1

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

    
class MT5Handler:
    @staticmethod
    def initialize_mt5() -> bool:
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            logger.info("MT5 initialized successfully")
            return True
        except Exception as e:
            logger.error(f"MT5 initialization error: {str(e)}")
            return False

    @staticmethod
    def get_historical_data(symbol: str, timeframe: int, days_back: int) -> Optional[list]:
        try:
            bars_per_day = {
                mt5.TIMEFRAME_M1: 1440,
                mt5.TIMEFRAME_M5: 288,
                mt5.TIMEFRAME_M15: 96,
                mt5.TIMEFRAME_M30: 48,
                mt5.TIMEFRAME_H1: 24,
                mt5.TIMEFRAME_H4: 6,
                mt5.TIMEFRAME_D1: 1
            }
            
            num_bars = days_back * bars_per_day.get(timeframe, 96)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
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
                'Volume_ratio': volumes / np.convolve(volumes, np.ones(20)/20, mode='same'),
                'time': [bar.time for bar in bars]
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
    def generate_signals(indicators: dict) -> np.ndarray:
        try:
            signals = np.zeros(len(indicators['close']))
            
            # Simple trend following conditions
            buy_conditions = (
                (indicators['EMA_fast'] > indicators['EMA_medium']) &
                (indicators['EMA_medium'] > indicators['EMA_slow']) &
                (indicators['RSI'] > TradingConfig.RSI_OVERSOLD) & 
                (indicators['RSI'] < 60) &  # More conservative RSI upper bound for buys
                (indicators['MACD_hist'] > 0) &
                (indicators['Volume_ratio'] > TradingConfig.MIN_VOLUME_RATIO)
            )
            
            sell_conditions = (
                (indicators['EMA_fast'] < indicators['EMA_medium']) &
                (indicators['EMA_medium'] < indicators['EMA_slow']) &
                (indicators['RSI'] < TradingConfig.RSI_OVERBOUGHT) & 
                (indicators['RSI'] > 40) &  # More conservative RSI lower bound for sells
                (indicators['MACD_hist'] < 0) &
                (indicators['Volume_ratio'] > TradingConfig.MIN_VOLUME_RATIO)
            )
            
            signals[buy_conditions] = 1
            signals[sell_conditions] = -1
            
            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

class Backtester:
    @staticmethod
    def calculate_position_size(indicators: dict, index: int, capital: float, current_price: float) -> float:
        atr = indicators['ATR'][index]
        stop_loss_distance = atr * TradingConfig.STOP_LOSS_ATR_MULT
        risk_amount = capital * TradingConfig.MAX_RISK_PER_TRADE
        position_size = risk_amount / stop_loss_distance
        return min(position_size, capital * TradingConfig.POSITION_SIZE)

    @staticmethod
    def backtest_strategy(indicators: dict, signals: np.ndarray, initial_capital: float = TradingConfig.INITIAL_CAPITAL) -> Tuple[dict, Dict]:
        try:
            equity_curve = np.full_like(indicators['close'], initial_capital)
            stop_losses = indicators['ATR'] * TradingConfig.STOP_LOSS_ATR_MULT
            take_profits = indicators['ATR'] * TradingConfig.TAKE_PROFIT_ATR_MULT
            
            capital = initial_capital
            position = 0
            entry_price = 0
            trades = []
            
            for i in range(1, len(indicators['close'])):
                current_price = indicators['close'][i]
                
                if signals[i] != 0 and position == 0:
                    position = signals[i]
                    entry_price = current_price
                    position_size = Backtester.calculate_position_size(
                        indicators, i, capital, entry_price
                    )
                    stop_loss = entry_price - (position * stop_losses[i])
                    take_profit = entry_price + (position * take_profits[i])
                    
                elif position != 0:
                    if position == 1:
                        if current_price <= stop_loss:
                            capital *= (1 - TradingConfig.MAX_RISK_PER_TRADE)
                            trades.append({'type': 'loss', 'return': -TradingConfig.MAX_RISK_PER_TRADE})
                            position = 0
                        elif current_price >= take_profit:
                            capital *= (1 + (TradingConfig.MAX_RISK_PER_TRADE * 2))
                            trades.append({'type': 'win', 'return': TradingConfig.MAX_RISK_PER_TRADE * 2})
                            position = 0
                    else:
                        if current_price >= stop_loss:
                            capital *= (1 - TradingConfig.MAX_RISK_PER_TRADE)
                            trades.append({'type': 'loss', 'return': -TradingConfig.MAX_RISK_PER_TRADE})
                            position = 0
                        elif current_price <= take_profit:
                            capital *= (1 + (TradingConfig.MAX_RISK_PER_TRADE * 2))
                            trades.append({'type': 'win', 'return': TradingConfig.MAX_RISK_PER_TRADE * 2})
                            position = 0
                            
                equity_curve[i] = capital
            
            indicators['equity'] = equity_curve
            metrics = Backtester.calculate_metrics(equity_curve, trades, initial_capital)
            return indicators, metrics

        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            raise

    @staticmethod
    def calculate_metrics(equity: np.ndarray, trades: list, initial_capital: float) -> Dict:
        returns = np.diff(equity) / equity[:-1]
        winning_trades = len([t for t in trades if t['type'] == 'win'])
        total_trades = len(trades)
        
        # Calculate max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)
        
        metrics = {
            'Total Returns': (equity[-1] - initial_capital) / initial_capital,
            'Sharpe Ratio': np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 0 else 0,
            'Max Drawdown': max_drawdown,
            'Win Rate': winning_trades / total_trades if total_trades > 0 else 0,
            'Total Trades': total_trades
        }
        return metrics

class Visualizer:
    @staticmethod
    def plot_results(indicators: dict, signals: np.ndarray):
        try:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
            
            times = indicators['time']
            
            # Price and EMAs
            ax1.plot(times, indicators['close'], label='Price', alpha=0.7)
            ax1.plot(times, indicators['EMA_fast'], label=f'EMA{TradingConfig.EMA_FAST}')
            ax1.plot(times, indicators['EMA_medium'], label=f'EMA{TradingConfig.EMA_MEDIUM}')
            ax1.plot(times, indicators['EMA_slow'], label=f'EMA{TradingConfig.EMA_SLOW}')
            
            # Signals
            buy_signals = signals == 1
            sell_signals = signals == -1
            
            ax1.scatter(
                [times[i] for i in range(len(times)) if buy_signals[i]], 
                [indicators['close'][i] for i in range(len(times)) if buy_signals[i]],
                marker='^', color='g', label='Buy Signal', s=100
            )
            ax1.scatter(
                [times[i] for i in range(len(times)) if sell_signals[i]], 
                [indicators['close'][i] for i in range(len(times)) if sell_signals[i]],
                marker='v', color='r', label='Sell Signal', s=100
            )
            
            ax1.set_title('Price Action and Signals')
            ax1.legend()
            ax1.grid(True)
            
            # RSI
            ax2.plot(times, indicators['RSI'], color='purple', label='RSI')
            ax2.axhline(y=TradingConfig.RSI_OVERBOUGHT, color='r', linestyle='--')
            ax2.axhline(y=TradingConfig.RSI_OVERSOLD, color='g', linestyle='--')
            ax2.set_title('RSI Indicator')
            ax2.legend()
            ax2.grid(True)
            
            # MACD
            ax3.plot(times, indicators['MACD_line'], label='MACD Line')
            ax3.plot(times, indicators['MACD_signal'], label='Signal Line')
            ax3.bar(times, indicators['MACD_hist'], label='MACD Histogram', alpha=0.3)
            ax3.set_title('MACD')
            ax3.legend()
            ax3.grid(True)
            
            # Equity curve
            ax4.plot(times, indicators['equity'], label='Portfolio Value')
            ax4.set_title('Portfolio Equity Curve')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise

def main():
    try:
        if not MT5Handler.initialize_mt5():
            return
        
        bars = MT5Handler.get_historical_data(
            TradingConfig.SYMBOL,
            TradingConfig.TIMEFRAME,
            TradingConfig.DAYS_TO_ANALYZE
        )
        
        if bars is None:
            return
        
        indicators = TechnicalIndicators.calculate_indicators(bars)
        signals = SignalGenerator.generate_signals(indicators)
        indicators, metrics = Backtester.backtest_strategy(indicators, signals)
        
        logger.info("\nBacktest Results:")
        logger.info(f"Total Returns: {metrics['Total Returns']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        logger.info(f"Maximum Drawdown: {metrics['Max Drawdown']:.2%}")
        logger.info(f"Win Rate: {metrics['Win Rate']:.2%}")
        logger.info(f"Total Trades: {metrics['Total Trades']}")
        
        Visualizer.plot_results(indicators, signals)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()