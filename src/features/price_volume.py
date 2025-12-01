"""
Price and volume feature engineering.
"""

import pandas as pd
import numpy as np


class PriceVolumeFeatures:
    """Extract price and volume features."""

    def compute_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Compute price and volume features."""
        # Handle multi-level columns
        if isinstance(ohlcv.columns, pd.MultiIndex):
            close = ohlcv['Close'].iloc[:, 0]
            high = ohlcv['High'].iloc[:, 0]
            low = ohlcv['Low'].iloc[:, 0]
            open_price = ohlcv['Open'].iloc[:, 0]
            volume = ohlcv['Volume'].iloc[:, 0]
        else:
            close = ohlcv['Close'] if 'Close' in ohlcv.columns else ohlcv['Adj Close']
            high = ohlcv['High']
            low = ohlcv['Low']
            open_price = ohlcv['Open']
            volume = ohlcv['Volume']

        features = pd.DataFrame(index=ohlcv.index)

        # Intraday range
        features['intraday_range'] = (high - low) / (close + 1e-8)
        features['open_close_diff'] = (close - open_price) / (open_price + 1e-8)

        # Volume features
        features['log_volume'] = np.log(volume + 1)
        features['volume_change'] = volume.pct_change()

        return features
