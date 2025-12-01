"""
Feature engineering pipeline orchestration.
"""

import pandas as pd
from typing import Dict, Optional
from .technical import TechnicalFeatures
from .price_volume import PriceVolumeFeatures
from .market_context import MarketContextFeatures


class FeaturePipeline:
    """Orchestrates all feature engineering."""

    def __init__(
        self,
        rolling_windows=[5, 10, 20, 60],
        return_lags=[1, 2, 3, 4, 5, 10, 20]
    ):
        self.technical = TechnicalFeatures(rolling_windows, return_lags)
        self.price_volume = PriceVolumeFeatures()
        self.market_context = MarketContextFeatures()

    def create_features(
        self,
        ohlcv: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Create all features.

        Args:
            ohlcv: OHLCV data for main ticker
            market_data: Optional market context data

        Returns:
            DataFrame with all features
        """
        # Technical features
        technical_features = self.technical.compute_all_features(ohlcv)

        # Price/volume features
        pv_features = self.price_volume.compute_features(ohlcv)

        # Combine
        all_features = pd.concat([technical_features, pv_features], axis=1)

        # Market context features if provided
        if market_data:
            market_features = self.market_context.compute_features(
                market_data,
                ohlcv.index
            )
            all_features = pd.concat([all_features, market_features], axis=1)

        return all_features
