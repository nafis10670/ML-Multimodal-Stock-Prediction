"""
Data preprocessing and cleaning module.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses and aligns stock market data."""

    def __init__(
        self,
        fill_method: str = "ffill",
        max_consecutive_nans: int = 5,
        drop_threshold: float = 0.5
    ):
        self.fill_method = fill_method
        self.max_consecutive_nans = max_consecutive_nans
        self.drop_threshold = drop_threshold

    def align_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        reference_ticker: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """Align multiple time series to common dates."""
        if not data_dict:
            return {}

        if reference_ticker and reference_ticker in data_dict:
            common_index = data_dict[reference_ticker].index
        else:
            indices = [df.index for df in data_dict.values()]
            common_index = indices[0]
            for idx in indices[1:]:
                common_index = common_index.intersection(idx)

        aligned_data = {}
        for ticker, df in data_dict.items():
            aligned_df = df.reindex(common_index)
            aligned_data[ticker] = aligned_df

        return aligned_data

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in a DataFrame."""
        df_clean = df.copy()

        missing_fraction = df_clean.isnull().sum() / len(df_clean)
        columns_to_drop = missing_fraction[missing_fraction > self.drop_threshold].index
        if len(columns_to_drop) > 0:
            logger.warning(f"Dropping columns with >{self.drop_threshold*100}% missing")
            df_clean = df_clean.drop(columns=columns_to_drop)

        if self.fill_method == "ffill":
            df_clean = df_clean.fillna(method='ffill', limit=self.max_consecutive_nans)
        elif self.fill_method == "bfill":
            df_clean = df_clean.fillna(method='bfill', limit=self.max_consecutive_nans)

        df_clean = df_clean.dropna()
        return df_clean
