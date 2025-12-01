"""
Linear regression baselines.
"""

from sklearn.linear_model import Ridge, Lasso, LinearRegression
import numpy as np


class RidgePredictor:
    """Ridge regression baseline."""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LassoPredictor:
    """Lasso regression baseline."""

    def __init__(self, alpha=0.01):
        self.model = Lasso(alpha=alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
