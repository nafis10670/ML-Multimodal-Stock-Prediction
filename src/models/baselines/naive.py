"""
Naive baseline models.
"""

import numpy as np


class NaivePredictor:
    """Random walk baseline - predicts zero return."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class MomentumPredictor:
    """Momentum baseline - predicts last return."""

    def __init__(self, lag=1):
        self.lag = lag

    def fit(self, X, y):
        return self

    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values
        if X.shape[1] > self.lag - 1:
            return X[:, self.lag - 1]
        return np.zeros(len(X))
