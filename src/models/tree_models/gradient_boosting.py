"""
Gradient boosting models (XGBoost/LightGBM).
"""

import xgboost as xgb
import lightgbm as lgb
import numpy as np


class XGBoostPredictor:
    """XGBoost regression model."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        **kwargs
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LightGBMPredictor:
    """LightGBM regression model."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        **kwargs
    ):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbose=-1,
            **kwargs
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
