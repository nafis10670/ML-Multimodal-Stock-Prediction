"""
Walk-forward validation for time series models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesSplit:
    """Represents a single train/validation/test split."""
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime

    def __repr__(self):
        return (
            f"Split(train: {self.train_start.date()} to {self.train_end.date()}, "
            f"val: {self.val_start.date()} to {self.val_end.date()}, "
            f"test: {self.test_start.date()} to {self.test_end.date()})"
        )


class WalkForwardValidator:
    """Walk-forward validation for time series data."""

    def __init__(
        self,
        train_years: int = 5,
        val_years: int = 1,
        test_years: int = 1,
        step_months: int = 12,
        embargo_days: int = 5
    ):
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.step_months = step_months
        self.embargo_days = embargo_days

    def generate_splits(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[TimeSeriesSplit]:
        """Generate walk-forward splits."""
        splits = []

        current_test_start = start_date + timedelta(
            days=365 * (self.train_years + self.val_years)
        )

        while True:
            test_end = current_test_start + timedelta(days=365 * self.test_years)

            if test_end > end_date:
                break

            val_end = current_test_start - timedelta(days=1)
            val_start = val_end - timedelta(days=365 * self.val_years) + timedelta(days=1)

            train_end = val_start - timedelta(days=self.embargo_days + 1)
            train_start = train_end - timedelta(days=365 * self.train_years) + timedelta(days=1)

            if train_start < start_date:
                train_start = start_date

            split = TimeSeriesSplit(
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                val_end=val_end,
                test_start=current_test_start,
                test_end=test_end
            )

            splits.append(split)
            current_test_start += timedelta(days=30 * self.step_months)

        return splits

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        split: TimeSeriesSplit
    ) -> Tuple:
        """Split features and target for a given split."""
        train_mask = (X.index >= split.train_start) & (X.index <= split.train_end)
        val_mask = (X.index >= split.val_start) & (X.index <= split.val_end)
        test_mask = (X.index >= split.test_start) & (X.index <= split.test_end)

        return (
            X.loc[train_mask], y.loc[train_mask],
            X.loc[val_mask], y.loc[val_mask],
            X.loc[test_mask], y.loc[test_mask]
        )


class ExpandingWindowScaler:
    """Scaler that fits only on training data."""

    def __init__(self, method: str = "standard"):
        self.method = method
        self.params = {}

    def fit(self, X: pd.DataFrame):
        if self.method == "standard":
            self.params['mean'] = X.mean()
            self.params['std'] = X.std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method == "standard":
            return (X - self.params['mean']) / (self.params['std'] + 1e-8)
        return X

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
