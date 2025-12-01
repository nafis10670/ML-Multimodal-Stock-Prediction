"""
Market context features (VIX, sector ETFs, macro).
"""

import pandas as pd
import numpy as np
from typing import Dict


class MarketContextFeatures:
    """Extract market context features."""

    def compute_features(
        self,
        market_data: Dict[str, pd.DataFrame],
        target_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """
        Compute market context features.

        Args:
            market_data: Dictionary of market indicator data
            target_index: Index to align features to

        Returns:
            DataFrame with market context features
        """
        features = pd.DataFrame(index=target_index)

        for ticker, df in market_data.items():
            # Handle multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                close = df['Close'].iloc[:, 0]
            else:
                close = df['Close'] if 'Close' in df.columns else df['Adj Close']

            # Align to target index
            close = close.reindex(target_index).fillna(method='ffill')

            # Ticker name for features
            clean_ticker = ticker.replace('^', '').replace('-', '_').replace('.', '_')

            # Level
            features[f'{clean_ticker}_close'] = close

            # Returns
            features[f'{clean_ticker}_return'] = np.log(close / close.shift(1))

            # Moving averages
            features[f'{clean_ticker}_ma_20'] = close.rolling(20).mean()
            features[f'{clean_ticker}_ma_60'] = close.rolling(60).mean()

        return features
