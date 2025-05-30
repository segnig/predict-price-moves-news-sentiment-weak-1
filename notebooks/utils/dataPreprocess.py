import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from IPython.display import display, Markdown
from typing import Optional, Dict, List, Tuple, Union



import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional



class DataLoader:
    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with a file path.
        
        Args:
            file_path (str): Path to the CSV file containing stock data.
        """
        self.file_path = file_path
        self.required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        self.optional_columns = ['Dividends', 'Stock Splits']
        self.data = self.load_data()

    def load_data(self) -> pd.DataFrame:
        """
        Load stock data from a CSV file with enhanced validation.
        
        Returns:
            pd.DataFrame: DataFrame containing stock data.
        """
        try:
            data = pd.read_csv(self.file_path)
            
            # Check required columns
            missing_required = [col for col in self.required_columns if col not in data.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {', '.join(missing_required)}")
                
            # Convert date and sort
            data['Date'] = pd.to_datetime(data['Date'])
            data.sort_values('Date', inplace=True)
            
            # Check for duplicates
            if data.duplicated(subset=['Date']).any():
                print("Warning: Duplicate dates found. Keeping first occurrence.")
                data.drop_duplicates(subset=['Date'], keep='first', inplace=True)
                
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def missing_data_report(self) -> pd.DataFrame:
        """
        Generate comprehensive missing data report.
        
        Returns:
            pd.DataFrame: Report showing missing data statistics.
        """
        if self.data.empty:
            return pd.DataFrame()
            
        report = pd.DataFrame({
            'Missing Values': self.data.isnull().sum(),
            'Missing Percentage': (self.data.isnull().mean() * 100).round(2),
            'Data Type': self.data.dtypes
        })
        
        # Highlight critical columns
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        report['Critical'] = report.index.map(lambda x: 'Yes' if x in critical_cols else 'No')
        
        # Display report
        display(Markdown("### Missing Data Report"))
        
        if report['Missing Values'].sum() == 0:
            display(Markdown("* **No missing data found.**"))
        else:
            display(report[report['Missing Values'] > 0])
            
        return report

    def get_columns_summary(self) -> dict:
        """
        Get detailed column summary including data types and value ranges.
        
        Returns:
            dict: Column metadata summary.
        """
        if self.data.empty:
            return {}
            
        summary = {}
        for col in self.data.columns:
            col_info = {
                'dtype': str(self.data[col].dtype),
                'unique_count': self.data[col].nunique(),
                'missing': self.data[col].isnull().sum()
            }
            
            if pd.api.types.is_numeric_dtype(self.data[col]):
                col_info.update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'median': self.data[col].median()
                })
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                col_info['range'] = (self.data[col].min(), self.data[col].max())
                
            summary[col] = col_info
            
        return summary

    def data_health_check(self) -> dict:
        """
        Perform comprehensive data quality check.
        
        Returns:
            dict: Data health metrics.
        """
        if self.data.empty:
            return {}
            
        # Safely check for price columns
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close'] 
                     if col in self.data.columns]
        
        health = {
            'start_date': self.data['Date'].min(),
            'end_date': self.data['Date'].max(),
            'total_days': (self.data['Date'].max() - self.data['Date'].min()).days,
            'trading_days': len(self.data),
            'completeness': len(self.data) / ((self.data['Date'].max() - self.data['Date'].min()).days + 1),
            'zero_volume_days': (self.data['Volume'] == 0).sum() if 'Volume' in self.data.columns else 0,
            'negative_price_days': ((self.data[price_cols] < 0).any(axis=1).sum() if price_cols else 0)
        }
        
        # Generate report
        report = f"""
### Data Health Report
- **Date Range**: {health['start_date'].strftime('%Y-%m-%d')} to {health['end_date'].strftime('%Y-%m-%d')}
- **Total Calendar Days**: {health['total_days'] + 1}
- **Trading Days**: {health['trading_days']} ({health['completeness']:.2%} coverage)
- **Missing Dates**: {health['total_days'] + 1 - health['trading_days']}
- **Zero Volume Days**: {health['zero_volume_days']}
"""
        
        # Only add price info if we have price columns
        if price_cols:
            report += f"- **Days with Negative Prices**: {health['negative_price_days']}"
        
        # Check for data anomalies
        if price_cols and health['negative_price_days'] > 0:
            report += "\n\n⚠️ **Warning**: Negative prices detected. Data may contain errors."
        if 'Volume' in self.data.columns and health['zero_volume_days'] > 10:
            report += f"\n\n⚠️ **Warning**: High number of zero-volume days ({health['zero_volume_days']})."
        if health['completeness'] < 0.7:
            report += f"\n\n⚠️ **Warning**: Low date coverage ({health['completeness']:.2%}). Significant missing dates."
        
        display(Markdown(report))
        return health

    def get_price_statistics(self) -> pd.DataFrame:
        """
        Calculate price statistics for available OHLC and Adjusted Close columns.
        
        Returns:
            pd.DataFrame: Statistical summary of price columns.
        """
        # Safely get available price columns
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close', 'Adj Close'] 
                     if col in self.data.columns]
        
        if not price_cols:
            display(Markdown("### Price Statistics\n*No price columns available in the dataset.*"))
            return pd.DataFrame()
            
        stats = self.data[price_cols].describe().T
        
        # Calculate volatility for available columns
        if 'Close' in self.data.columns:
            returns = self.data['Close'].pct_change()
            stats.loc['Close', 'volatility'] = returns.std() * np.sqrt(252) if not returns.empty else np.nan
        elif 'Adj Close' in self.data.columns:
            returns = self.data['Adj Close'].pct_change()
            stats.loc['Adj Close', 'volatility'] = returns.std() * np.sqrt(252) if not returns.empty else np.nan
        
        display(Markdown("### Price Statistics"))
        display(stats)
        return stats

    def get_volume_analysis(self) -> dict:
        """
        Analyze trading volume patterns.
        
        Returns:
            dict: Volume statistics and insights.
        """
        if 'Volume' not in self.data.columns:
            display(Markdown("### Volume Analysis\n*'Volume' column not found in dataset.*"))
            return {}
            
        analysis = {
            'avg_volume': self.data['Volume'].mean(),
            'max_volume': self.data['Volume'].max(),
            'min_volume': self.data['Volume'].min(),
            'volume_volatility': self.data['Volume'].pct_change().std(),
            'high_volume_days': (self.data['Volume'] > 1.5 * self.data['Volume'].mean()).sum()
        }
        
        # Generate report
        report = f"""
### Volume Analysis
- **Average Daily Volume**: {analysis['avg_volume']:,.0f}
- **Peak Volume**: {analysis['max_volume']:,.0f}
- **Lowest Volume**: {analysis['min_volume']:,.0f}
- **High Volume Days (>1.5x avg)**: {analysis['high_volume_days']}
"""
        display(Markdown(report))
        
        # Monthly volume analysis if we have enough data
        if len(self.data) > 60:  # At least 2 months
            monthly_volume = self.data.set_index('Date')['Volume'].resample('M').sum()
            monthly_volume.plot(kind='bar', title='Monthly Volume Trend', figsize=(12, 4))
        
        return analysis

    def get_corporate_actions_summary(self) -> dict:
        """
        Summarize dividends and stock splits.
        
        Returns:
            dict: Corporate actions summary.
        """
        actions = {}
        report = "### Corporate Actions Summary\n"
        
        if 'Dividends' in self.data.columns:
            dividend_days = (self.data['Dividends'] > 0).sum()
            total_dividends = self.data['Dividends'].sum()
            actions['dividends'] = {
                'count': dividend_days,
                'total': total_dividends,
                'avg': total_dividends / dividend_days if dividend_days > 0 else 0
            }
            report += f"- **Dividend Payments**: {actions['dividends']['count']} days\n"
            report += f"- **Total Dividends**: ${actions['dividends']['total']:.4f} per share\n"
            report += f"- **Average Dividend**: ${actions['dividends']['avg']:.4f} per payment\n"
        
        if 'Stock Splits' in self.data.columns:
            split_days = (self.data['Stock Splits'] > 0).sum()
            actions['splits'] = {
                'count': split_days,
                'last_split': self.data.loc[self.data['Stock Splits'] > 0, 'Date'].max() 
                if split_days > 0 else None
            }
            report += f"- **Stock Splits**: {actions['splits']['count']} occurrences\n"
            if actions['splits']['last_split']:
                report += f"- **Last Split**: {actions['splits']['last_split'].strftime('%Y-%m-%d')}\n"
        
        if not actions:
            report += "No corporate action data available."
            
        display(Markdown(report))
        return actions

    def get_news_impact_analysis(self, date: str, window: int = 30) -> pd.DataFrame:
        """
        Enhanced news impact analysis with technical context.
        
        Args:
            date (str): Event date in 'YYYY-MM-DD' format
            window (int): Analysis window in days (default 30)
            
        Returns:
            pd.DataFrame: Analysis data with technical features
        """
        try:
            event_date = pd.to_datetime(date)
            start_date = event_date - timedelta(days=window)
            end_date = event_date + timedelta(days=window)
            
            # Filter data
            mask = (self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)
            analysis_data = self.data.loc[mask].copy()
            
            if analysis_data.empty:
                display(Markdown(f"### News Impact Analysis: {date}\n*No data found for {start_date} to {end_date}*"))
                return pd.DataFrame()
            
            # Add technical features if we have Close prices
            if 'Close' in analysis_data.columns:
                analysis_data['Daily_Return'] = analysis_data['Close'].pct_change()
                analysis_data['Cumulative_Return'] = (1 + analysis_data['Daily_Return']).cumprod() - 1
                
                if len(analysis_data) > 20:  # Only add SMA if enough data
                    analysis_data['SMA_20'] = analysis_data['Close'].rolling(20).mean()
                
                if len(analysis_data) > 10:  # Only add volatility if enough data
                    analysis_data['Volatility'] = analysis_data['Daily_Return'].rolling(10).std() * np.sqrt(252)
            
            # Flag event date
            analysis_data['Event_Day'] = analysis_data['Date'] == event_date
            
            # Generate report
            display(Markdown(f"### News Impact Analysis: {date}"))
            
            # Plot cumulative returns if available
            if 'Cumulative_Return' in analysis_data.columns:
                ax = analysis_data.plot(x='Date', y='Cumulative_Return', 
                                      title=f'Cumulative Returns {window} Days Around Event',
                                      figsize=(12, 4))
                ax.axvline(x=event_date, color='r', linestyle='--', label='Event Day')
            
            return analysis_data
            
        except Exception as e:
            display(Markdown(f"### News Impact Analysis Error\n*{str(e)}*"))
            return pd.DataFrame()
        
        

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataPreprocessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            if 'Date' in self.data.columns:
                self.data['Date'] = pd.to_datetime(self.data['Date'])
                self.data.set_index('Date', inplace=True)
            else:
                self.data.index = pd.to_datetime(self.data.index)
        self.data.sort_index(inplace=True)
        self.standardize_columns()

    def standardize_columns(self):
        self.data.columns = [col.lower().replace(' ', '_') for col in self.data.columns]
        if 'adj_close' in self.data.columns:
            self.data.rename(columns={'adj_close': 'adjusted_close'}, inplace=True)

    def handle_missing_data(self, method: str = 'ffill') -> None:
        valid_methods = ['ffill', 'bfill', 'interpolate']
        if method not in valid_methods:
            raise ValueError(f"Invalid method. Choose from {valid_methods}")
        
        if method == 'interpolate':
            self.data.interpolate(method='linear', inplace=True)
        else:
            self.data.fillna(method=method, inplace=True)
            
        if 'volume' in self.data.columns:
            self.data['volume'].fillna(0, inplace=True)

    def add_technical_indicators(self) -> None:
        self._add_moving_averages()
        self._add_rsi()
        self._add_macd()
        self._add_bollinger_bands()
        self._add_atr()
        
        if 'volume' in self.data.columns:
            self._add_volume_indicators()
            
        self._add_price_features()
        self._add_time_features()

    def _add_moving_averages(self, windows: list = [5, 20, 50, 200]) -> None:
        close_col = 'adjusted_close' if 'adjusted_close' in self.data.columns else 'close'
        
        for window in windows:
            self.data[f'sma_{window}'] = self.data[close_col].rolling(window).mean()
            self.data[f'ema_{window}'] = self.data[close_col].ewm(span=window, adjust=False).mean()

    def _add_rsi(self, window: int = 14) -> None:
        close_col = 'adjusted_close' if 'adjusted_close' in self.data.columns else 'close'
        delta = self.data[close_col].diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def _add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        close_col = 'adjusted_close' if 'adjusted_close' in self.data.columns else 'close'
        
        ema_fast = self.data[close_col].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data[close_col].ewm(span=slow, adjust=False).mean()
        
        self.data['macd'] = ema_fast - ema_slow
        self.data['macd_signal'] = self.data['macd'].ewm(span=signal, adjust=False).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']

    def _add_bollinger_bands(self, window: int = 20, num_std: int = 2) -> None:
        close_col = 'adjusted_close' if 'adjusted_close' in self.data.columns else 'close'
        
        sma = self.data[close_col].rolling(window).mean()
        std = self.data[close_col].rolling(window).std()
        
        self.data['bollinger_mid'] = sma
        self.data['bollinger_upper'] = sma + (std * num_std)
        self.data['bollinger_lower'] = sma - (std * num_std)
        self.data['bollinger_%b'] = (
            (self.data[close_col] - self.data['bollinger_lower']) / 
            (self.data['bollinger_upper'] - self.data['bollinger_lower'])
        )

    def _add_atr(self, window: int = 14) -> None:
        if not all(col in self.data.columns for col in ['high', 'low', 'close']):
            return
            
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['atr'] = true_range.rolling(window).mean()

    def _add_volume_indicators(self, windows: list = [5, 20]) -> None:
        for window in windows:
            self.data[f'vol_ma_{window}'] = self.data['volume'].rolling(window).mean()
            self.data[f'vwap_{window}'] = (
                (self.data['volume'] * self.data['close']).rolling(window).sum() / 
                self.data['volume'].rolling(window).sum()
            )
        
        self.data['vol_roc'] = self.data['volume'].pct_change(periods=1) * 100

    def _add_price_features(self) -> None:
        if all(col in self.data.columns for col in ['high', 'low']):
            self.data['daily_range'] = self.data['high'] - self.data['low']
            self.data['range_pct'] = self.data['daily_range'] / self.data['low']
        
        close_col = 'adjusted_close' if 'adjusted_close' in self.data.columns else 'close'
        
        self.data['daily_return'] = self.data[close_col].pct_change()
        self.data['intraday_change'] = self.data['close'] - self.data['open']
        self.data['intraday_pct'] = self.data['intraday_change'] / self.data['open']
        self.data['volatility_30d'] = self.data['daily_return'].rolling(30).std() * np.sqrt(252)

    def _add_time_features(self) -> None:
        self.data['day_of_week'] = self.data.index.dayofweek
        self.data['month'] = self.data.index.month
        self.data['quarter'] = self.data.index.quarter
        self.data['year'] = self.data.index.year
        self.data['is_month_start'] = self.data.index.is_month_start.astype(int)
        self.data['is_month_end'] = self.data.index.is_month_end.astype(int)
        self.data['is_quarter_start'] = self.data.index.is_quarter_start.astype(int)
        self.data['is_quarter_end'] = self.data.index.is_quarter_end.astype(int)
        self.data['is_year_start'] = self.data.index.is_year_start.astype(int)
        self.data['is_year_end'] = self.data.index.is_year_end.astype(int)

    def add_news_sentiment(self, news_data: pd.DataFrame) -> None:
        if 'date' not in news_data.columns or 'sentiment' not in news_data.columns:
            raise ValueError("News data must contain 'date' and 'sentiment' columns")
            
        news_data = news_data.copy()
        news_data['date'] = pd.to_datetime(news_data['date'])
        news_data.set_index('date', inplace=True)
        daily_sentiment = news_data['sentiment'].resample('D').mean().to_frame('sentiment')
        self.data = self.data.merge(daily_sentiment, left_index=True, right_index=True, how='left')
        self.data['sentiment'].fillna(0, inplace=True)

    def prepare_for_modeling(self, target: str = 'close', lookahead: int = 1) -> pd.DataFrame:
        close_col = 'adjusted_close' if 'adjusted_close' in self.data.columns else 'close'
        self.data['target'] = self.data[close_col].shift(-lookahead)
        processed = self.data.dropna()
        cols_to_drop = ['dividends', 'stock_splits', 'capital_gains']
        for col in cols_to_drop:
            if col in processed.columns:
                processed.drop(columns=col, inplace=True)
        if 'target' in processed.columns:
            target_col = processed.pop('target')
            processed['target'] = target_col
        return processed

    def get_processed_data(self) -> pd.DataFrame:
        return self.data

class StockAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self._info = None
        self._dividends = None
        self._splits = None
        self._financials = None
        self._balance_sheet = None
        self._cashflow = None
        self._recommendations = None
        self._news = None

    @property
    def info(self) -> Dict:
        if self._info is None:
            try:
                self._info = self.stock.info
            except:
                self._info = {}
        return self._info

    @property
    def dividends(self) -> pd.Series:
        if self._dividends is None:
            self._dividends = self.stock.dividends
        return self._dividends

    @property
    def splits(self) -> pd.Series:
        if self._splits is None:
            self._splits = self.stock.splits
        return self._splits

    @property
    def financials(self) -> pd.DataFrame:
        if self._financials is None:
            try:
                self._financials = self.stock.financials
            except:
                self._financials = pd.DataFrame()
        return self._financials

    @property
    def balance_sheet(self) -> pd.DataFrame:
        if self._balance_sheet is None:
            try:
                self._balance_sheet = self.stock.balance_sheet
            except:
                self._balance_sheet = pd.DataFrame()
        return self._balance_sheet

    @property
    def cashflow(self) -> pd.DataFrame:
        if self._cashflow is None:
            try:
                self._cashflow = self.stock.cashflow
            except:
                self._cashflow = pd.DataFrame()
        return self._cashflow

    @property
    def recommendations(self) -> pd.DataFrame:
        if self._recommendations is None:
            try:
                self._recommendations = self.stock.recommendations
            except:
                self._recommendations = pd.DataFrame()
        return self._recommendations

    @property
    def news(self) -> List[Dict]:
        if self._news is None:
            try:
                self._news = self.stock.news
            except:
                self._news = []
        return self._news

    def get_historical_data(
        self,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
        prep: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        if start and end:
            data = self.stock.history(start=start, end=end, interval=interval)
        else:
            data = self.stock.history(period=period, interval=interval)
        
        if prep:
            preprocessor = DataPreprocessor(data.copy())
            preprocessor.handle_missing_data()
            preprocessor.add_technical_indicators()
            return data, preprocessor.get_processed_data()
        return data

    def get_news_impact_analysis(
        self,
        news_date: str,
        days_before: int = 5,
        days_after: int = 5,
        add_indicators: bool = True
    ) -> pd.DataFrame:
        start_date = (datetime.strptime(news_date, '%Y-%m-%d') - 
                     timedelta(days=days_before))
        end_date = (datetime.strptime(news_date, '%Y-%m-%d') + 
                   timedelta(days=days_after))

        data = self.get_historical_data(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        if add_indicators and not data.empty:
            preprocessor = DataPreprocessor(data.copy())
            preprocessor.handle_missing_data()
            preprocessor.add_technical_indicators()
            return preprocessor.get_processed_data()
        return data

    def plot_stock_data(
        self,
        data: pd.DataFrame,
        indicators: bool = True,
        volume: bool = True,
        news_events: Optional[Dict] = None
    ) -> None:
        fig = make_subplots(
            rows=2 if volume else 1,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3] if volume else [1],
            specs=[[{"secondary_y": True}]] + [[{"secondary_y": False}]] if volume else [[{"secondary_y": True}]]
        )

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1,
            secondary_y=False
        )

        if indicators:
            for col in data.columns:
                if col.startswith('sma_'):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[col],
                            name=col.upper(),
                            line=dict(width=1.5)
                        ),
                        row=1, col=1,
                        secondary_y=False
                    )
                elif col.startswith('ema_'):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[col],
                            name=col.upper(),
                            line=dict(width=1.5, dash='dot')
                        ),
                        row=1, col=1,
                        secondary_y=False
                    )
            
            if 'bollinger_upper' in data.columns and 'bollinger_lower' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['bollinger_upper'],
                        name='Bollinger Upper',
                        line=dict(color='rgba(200, 200, 200, 0.5)')
                    ),
                    row=1, col=1,
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['bollinger_lower'],
                        name='Bollinger Lower',
                        line=dict(color='rgba(200, 200, 200, 0.5)'),
                        fill='tonexty',
                        fillcolor='rgba(200, 200, 200, 0.1)'
                    ),
                    row=1, col=1,
                    secondary_y=False
                )
            
            if 'rsi' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['rsi'],
                        name='RSI',
                        line=dict(color='purple')
                    ),
                    row=2 if volume else 1, col=1,
                    secondary_y=True
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", 
                              row=2 if volume else 1, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", 
                              row=2 if volume else 1, col=1)

        if volume:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(100, 100, 200, 0.5)'
                ),
                row=2, col=1
            )
            
            for col in data.columns:
                if col.startswith('vol_ma_'):
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[col],
                            name=col.upper(),
                            line=dict(width=1.5)
                        ),
                        row=2, col=1
                    )

        if news_events:
            for date, headline in news_events.items():
                event_date = pd.to_datetime(date)
                if data.index.min() <= event_date <= data.index.max():
                    fig.add_vline(
                        x=event_date, 
                        line_width=2, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=headline[:20] + "..." if len(headline) > 20 else headline,
                        annotation_position="top right"
                    )

        fig.update_layout(
            title=f'{self.ticker} Stock Analysis',
            yaxis_title='Price',
            xaxis_title='Date',
            template='plotly_white',
            height=800,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_yaxes(title_text="Price", row=1, col=1)
        if volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            if indicators and 'rsi' in data.columns:
                fig.update_yaxes(title_text="RSI", row=2, col=1, secondary_y=True)
                fig.update_yaxes(range=[0, 100], row=2, col=1, secondary_y=True)
        
        if volume:
            fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
            fig.update_xaxes(rangeslider_visible=False, row=2, col=1)
        else:
            fig.update_xaxes(rangeslider_visible=True)

        fig.show()

    def analyze_news_impact(
        self,
        news_data: pd.DataFrame,
        window: int = 5
    ) -> pd.DataFrame:
        results = []
        
        for _, news in news_data.iterrows():
            news_date = news['date']
            try:
                event_data = self.get_news_impact_analysis(
                    news_date,
                    days_before=window,
                    days_after=window,
                    add_indicators=True
                )
                
                if not event_data.empty:
                    price_before = event_data['Close'].iloc[0]
                    price_after = event_data['Close'].iloc[-1]
                    price_change = ((price_after - price_before) / price_before) * 100
                    
                    volatility = event_data['Close'].pct_change().std() * 100
                    volume_change = ((event_data['Volume'].iloc[-1] - event_data['Volume'].iloc[0]) / 
                                   event_data['Volume'].iloc[0] * 100) if event_data['Volume'].iloc[0] != 0 else 0
                    
                    event_day = event_data.loc[pd.to_datetime(news_date)]
                    rsi = event_day.get('rsi', np.nan)
                    macd = event_day.get('macd', np.nan)
                    
                    results.append({
                        'date': news_date,
                        'headline': news['headline'],
                        'source': news.get('source', ''),
                        'sentiment': news.get('sentiment', 0),
                        'price_before': price_before,
                        'price_after': price_after,
                        'price_change_pct': price_change,
                        'volatility_pct': volatility,
                        'volume_change_pct': volume_change,
                        'rsi': rsi,
                        'macd': macd,
                        'event_data': event_data
                    })
                    
            except Exception as e:
                print(f"Error analyzing news on {news_date}: {str(e)}")
                continue
        
        return pd.DataFrame(results)

    def compare_with_index(
        self, 
        index_ticker: str = '^GSPC', 
        period: str = '1y'
    ) -> pd.DataFrame:
        stock_data = self.get_historical_data(period=period)
        if stock_data.empty:
            return pd.DataFrame()
        
        index = yf.Ticker(index_ticker)
        index_data = index.history(period=period)
        
        if index_data.empty:
            return pd.DataFrame()
        
        comparison = pd.DataFrame({
            self.ticker: stock_data['Close'].pct_change().cumsum(),
            index_ticker: index_data['Close'].pct_change().cumsum()
        }).dropna()
        
        return comparison

    def plot_correlation_heatmap(
        self, 
        tickers: List[str], 
        period: str = '1y'
    ) -> None:
        all_tickers = [self.ticker] + tickers
        data = {}
        for t in all_tickers:
            try:
                stock = yf.Ticker(t)
                hist = stock.history(period=period)
                if not hist.empty:
                    data[t] = hist['Close'].pct_change().dropna()
            except:
                continue
        
        if not data:
            print("No data available for correlation analysis.")
            return
        
        df = pd.DataFrame(data).dropna()
        corr = df.corr()
        
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title=f"Stock Correlation Heatmap ({period})"
        )
        fig.update_layout(height=600)
        fig.show()

    def calculate_risk_metrics(
        self, 
        data: Optional[pd.DataFrame] = None, 
        period: str = '1y'
    ) -> Dict:
        if data is None:
            data = self.get_historical_data(period=period)
        
        if data.empty:
            return {}
        
        returns = data['Close'].pct_change().dropna()
        
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
        
        beta = np.nan
        if len(returns) > 10:
            try:
                market = yf.Ticker('^GSPC').history(period=period)['Close'].pct_change().dropna()
                aligned_data = pd.concat([returns, market], axis=1).dropna()
                if len(aligned_data) > 2:
                    cov = np.cov(aligned_data.values.T)
                    beta = cov[0, 1] / cov[1, 1]
            except:
                pass
        
        return {
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'avg_daily_return': returns.mean(),
            'positive_days': (returns > 0).mean()
        }

    def plot_financials(
        self, 
        statement_type: str = 'income', 
        items: List[str] = ['Total Revenue', 'Net Income']
    ) -> None:
        if statement_type == 'income':
            data = self.financials
        elif statement_type == 'balance':
            data = self.balance_sheet
        elif statement_type == 'cashflow':
            data = self.cashflow
        else:
            print("Invalid statement type. Choose from 'income', 'balance', 'cashflow'")
            return
        
        if data.empty:
            print(f"No {statement_type} statement data available.")
            return
        
        plot_data = data.loc[data.index.isin(items)].T
        
        if plot_data.empty:
            print("None of the requested items found in the financial statement.")
            return
        
        fig = px.line(
            plot_data,
            title=f"{self.ticker} Financials",
            labels={'value': 'Amount (USD)', 'variable': 'Financial Item'},
            markers=True
        )
        fig.update_layout(
            hovermode='x unified',
            legend_title_text='Financial Items'
        )
        fig.show()

    def get_analyst_recommendations_summary(self) -> dict:
        if self.recommendations.empty:
            return {}
        
        recs = self.recommendations.copy()
        recs['Grading'] = recs['To Grade'].fillna(recs['Action'])
        
        rating_map = {
            'Buy': 5,
            'Strong Buy': 5,
            'Outperform': 4,
            'Overweight': 4,
            'Hold': 3,
            'Neutral': 3,
            'Market Perform': 3,
            'Sector Perform': 3,
            'Underperform': 2,
            'Underweight': 2,
            'Sell': 1,
            'Strong Sell': 1
        }
        
        recs['Rating'] = recs['Grading'].map(rating_map).fillna(3)
        recs['Date'] = recs.index
        
        recent = recs.sort_values('Date', ascending=False).head(10)
        distribution = recs['Rating'].value_counts().sort_index()
        avg_rating = recs['Rating'].mean()
        
        return {
            'recent_recommendations': recent,
            'rating_distribution': distribution,
            'average_rating': avg_rating
        }

    def plot_dividend_history(self) -> None:
        if self.dividends.empty:
            print("No dividend data available.")
            return
        
        fig = px.bar(
            self.dividends,
            x=self.dividends.index,
            y=self.dividends.values,
            title=f"{self.ticker} Dividend History",
            labels={'x': 'Date', 'y': 'Dividend per Share'}
        )
        fig.update_layout(bargap=0.2)
        fig.show()

# Main analysis workflow
if __name__ == "__main__":
    # Initialize StockAnalyzer for Microsoft
    msft_analyzer = StockAnalyzer("MSFT")
    
    # 1. Fetch and preprocess historical data
    msft_raw, msft_processed = msft_analyzer.get_historical_data(
        period="2y",
        prep=True
    )

    print("Processed Data Overview:")
    print(msft_processed[['Open', 'High', 'Low', 'Close', 'Volume', 'sma_20', 'rsi']].tail())

    # 2. Create news events DataFrame
    news_data = pd.DataFrame({
        'date': ['2023-01-18', '2023-04-25', '2023-07-25'],
        'headline': [
            'Microsoft announces 10,000 layoffs',
            'Azure growth slows amid cloud competition',
            'Microsoft beats Q4 earnings expectations'
        ],
        'source': ['WSJ', 'CNBC', 'Bloomberg'],
        'sentiment': [-0.85, -0.3, 0.9]
    })

    # 3. Analyze news impact
    news_impact = msft_analyzer.analyze_news_impact(news_data, window=5)

    print("\nNews Impact Analysis:")
    print(news_impact[['date', 'headline', 'price_change_pct', 'volatility_pct', 'rsi']])

    # 4. Prepare for machine learning
    preprocessor = DataPreprocessor(msft_raw.copy())
    preprocessor.handle_missing_data(method='ffill')
    preprocessor.add_technical_indicators()
    preprocessor.add_news_sentiment(news_data)
    model_data = preprocessor.prepare_for_modeling(target='close', lookahead=5)

    print("\nMachine Learning Dataset:")
    print(model_data[['sma_50', 'rsi', 'macd', 'vol_ma_5', 'sentiment', 'target']].tail())

    # 5. Perform comparative analysis
    competitors = ['AAPL', 'GOOG', 'AMZN', 'META', 'NVDA']
    comparison_data = pd.DataFrame()

    for ticker in [msft_analyzer.ticker] + competitors:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')['Close'].rename(ticker)
            comparison_data = pd.concat([comparison_data, hist], axis=1)
        except:
            continue

    # Calculate normalized returns
    normalized_returns = comparison_data.apply(
        lambda x: x / x.iloc[0] * 100
    ).dropna()

    print("\nNormalized Returns Comparison:")
    print(normalized_returns.tail())

    # 6. Risk analysis
    risk_metrics = {}
    for ticker in normalized_returns.columns:
        returns = normalized_returns[ticker].pct_change().dropna()
        risk_metrics[ticker] = {
            'volatility': returns.std() * np.sqrt(252),
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            'max_drawdown': (normalized_returns[ticker] / normalized_returns[ticker].cummax() - 1).min()
        }

    risk_df = pd.DataFrame(risk_metrics).T
    print("\nRisk Metrics Comparison:")
    print(risk_df)

    # 7. Correlation analysis
    correlation_matrix = normalized_returns.pct_change().corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

    # 8. Visualization
    # Create interactive charts
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price Comparison', 'Risk Metrics'),
        row_heights=[0.7, 0.3]
    )

    # Price comparison
    for ticker in normalized_returns.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized_returns.index,
                y=normalized_returns[ticker],
                name=ticker,
                mode='lines'
            ),
            row=1, col=1
        )

    # Volatility comparison
    fig.add_trace(
        go.Bar(
            x=risk_df.index,
            y=risk_df['volatility'],
            name='Volatility',
            marker_color='skyblue'
        ),
        row=2, col=1
    )

    fig.update_layout(
        title='Tech Stock Comparison (1Y)',
        height=700,
        showlegend=True
    )
    fig.show()

    # Correlation heatmap
    corr_fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title='Tech Stock Correlation (1Y)',
        labels=dict(color="Correlation")
    )
    corr_fig.update_layout(height=500)
    corr_fig.show()

    # News impact visualization
    if not news_impact.empty:
        fig = make_subplots(
            rows=len(news_impact), cols=1,
            subplot_titles=[f"{row.date}: {row.headline[:20]}..." for _, row in news_impact.iterrows()],
            vertical_spacing=0.05
        )
        
        for i, (_, event) in enumerate(news_impact.iterrows()):
            event_data = event['event_data']
            event_date = pd.to_datetime(event.date)
            
            # Price chart
            fig.add_trace(
                go.Scatter(
                    x=event_data.index,
                    y=event_data['Close'],
                    name='Price',
                    line=dict(color='blue')
                ),
                row=i+1, col=1
            )
            
            # Event marker
            fig.add_vline(
                x=event_date,
                line_width=2,
                line_dash="dash",
                line_color="red",
                row=i+1, col=1
            )
            
            # Volume chart
            if 'Volume' in event_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=event_data.index,
                        y=event_data['Volume'],
                        name='Volume',
                        marker_color='rgba(100, 100, 200, 0.5)',
                        yaxis='y2'
                    ),
                    row=i+1, col=1
                )
            
            # Update layout for this subplot
            fig.update_yaxes(title_text="Price", row=i+1, col=1)
            if 'Volume' in event_data.columns:
                fig.update_yaxes(title_text="Volume", row=i+1, col=1, secondary_y=True)
        
        fig.update_layout(
            title_text="News Event Impact Analysis",
            height=300 * len(news_impact),
            showlegend=False
        )
        fig.show()