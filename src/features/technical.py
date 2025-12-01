"""
Technical indicator feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List


class TechnicalFeatures:
    """Compute technical analysis features from OHLCV data."""

    def __init__(
        self,
        rolling_windows: List[int] = [5, 10, 20, 60],
        return_lags: List[int] = [1, 2, 3, 4, 5, 10, 20]
    ):
        self.rolling_windows = rolling_windows
        self.return_lags = return_lags

    def compute_returns(self, prices: pd.Series) -> pd.DataFrame:
        """Compute log returns and lagged returns."""
        features = pd.DataFrame(index=prices.index)

        log_returns = np.log(prices / prices.shift(1))
        features['log_return'] = log_returns
        features['simple_return'] = prices.pct_change()

        for lag in self.return_lags:
            features[f'return_lag_{lag}'] = log_returns.shift(lag)

        return features

    def compute_rolling_stats(self, returns: pd.Series) -> pd.DataFrame:
        """Compute rolling statistics of returns."""
        features = pd.DataFrame(index=returns.index)

        for window in self.rolling_windows:
            features[f'return_mean_{window}d'] = returns.rolling(window).mean()
            features[f'return_std_{window}d'] = returns.rolling(window).std()
            features[f'return_skew_{window}d'] = returns.rolling(window).skew()
            features[f'return_kurt_{window}d'] = returns.rolling(window).kurt()
            features[f'sharpe_{window}d'] = (
                features[f'return_mean_{window}d'] /
                (features[f'return_std_{window}d'] + 1e-8)
            )

        return features

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = prices.diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def compute_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """Compute MACD indicator."""
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal

        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_histogram': histogram
        })

    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Compute Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def compute_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """Compute Bollinger Bands."""
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        percent_b = (prices - lower) / (upper - lower + 1e-8)
        bandwidth = (upper - lower) / (middle + 1e-8)

        return pd.DataFrame({
            'bb_middle': middle,
            'bb_upper': upper,
            'bb_lower': lower,
            'bb_percent_b': percent_b,
            'bb_bandwidth': bandwidth
        })

    def compute_all_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features from OHLCV data."""
        # Handle multi-level columns from yfinance
        if isinstance(ohlcv.columns, pd.MultiIndex):
            close = ohlcv['Close'].iloc[:, 0] if ohlcv['Close'].ndim > 1 else ohlcv['Close']
            high = ohlcv['High'].iloc[:, 0] if ohlcv['High'].ndim > 1 else ohlcv['High']
            low = ohlcv['Low'].iloc[:, 0] if ohlcv['Low'].ndim > 1 else ohlcv['Low']
            volume = ohlcv['Volume'].iloc[:, 0] if ohlcv['Volume'].ndim > 1 else ohlcv['Volume']
        else:
            close = ohlcv['Close'] if 'Close' in ohlcv.columns else ohlcv['Adj Close']
            high = ohlcv['High']
            low = ohlcv['Low']
            volume = ohlcv['Volume']

        features = pd.DataFrame(index=ohlcv.index)

        # Returns
        return_features = self.compute_returns(close)
        features = pd.concat([features, return_features], axis=1)

        # Rolling statistics
        log_returns = return_features['log_return']
        rolling_features = self.compute_rolling_stats(log_returns)
        features = pd.concat([features, rolling_features], axis=1)

        # RSI
        features['rsi_14'] = self.compute_rsi(close, 14)

        # MACD
        macd_features = self.compute_macd(close)
        features = pd.concat([features, macd_features], axis=1)

        # ATR
        features['atr_14'] = self.compute_atr(high, low, close, 14)
        features['atr_normalized'] = features['atr_14'] / close

        # Bollinger Bands
        bb_features = self.compute_bollinger_bands(close)
        features = pd.concat([features, bb_features], axis=1)

        # Volume features
        features['volume_sma_20'] = volume.rolling(20).mean()
        features['volume_ratio'] = volume / (features['volume_sma_20'] + 1e-8)
        features['volume_zscore'] = (
            (volume - volume.rolling(20).mean()) /
            (volume.rolling(20).std() + 1e-8)
        )

        # Price momentum
        features['high_low_ratio'] = high / (low + 1e-8)
        features['close_to_high'] = close / (high + 1e-8)
        features['close_to_low'] = close / (low + 1e-8)

        return features
