import talib
import matplotlib.pyplot as plt
import pandas as pd


class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._prepare()

    def _prepare(self):
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        self.data.sort_index(inplace=True)
        self.fill_missing()
    
    def fill_missing(self):
        """Fill missing data using forward-fill then backward-fill."""
        self.data.ffill(inplace=True)
        self.data.bfill(inplace=True)

    def add_technical_indicators(self):
        """Add Moving Averages, RSI, MACD, Bollinger Bands."""
        df = self.data
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        macd, macdsignal, _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macdsignal
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        return df

    def get_processed_data(self):
        return self.data


class StockVisualizer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def plot_price_with_indicators(self):
        df = self.data
        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'], label='Close Price')
        plt.plot(df['SMA_20'], label='SMA 20')
        plt.plot(df['EMA_50'], label='EMA 50')
        plt.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='gray', alpha=0.1, label='Bollinger Bands')
        plt.title("Stock Price with Technical Indicators")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_rsi(self):
        plt.figure(figsize=(12, 4))
        plt.plot(self.data['RSI'], label='RSI')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title("Relative Strength Index (RSI)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_macd(self):
        plt.figure(figsize=(12, 4))
        plt.plot(self.data['MACD'], label='MACD')
        plt.plot(self.data['MACD_Signal'], label='Signal Line')
        plt.title("MACD and Signal Line")
        plt.legend()
        plt.grid(True)
        plt.show()
