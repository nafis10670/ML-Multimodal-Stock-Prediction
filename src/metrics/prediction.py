"""
Prediction metrics for model evaluation.
"""

import numpy as np
from typing import Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Information coefficient
    ic = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'ic': ic
    }


def compute_directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute directional prediction metrics."""
    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)

    accuracy = np.mean(true_direction == pred_direction)

    return {
        'directional_accuracy': accuracy
    }


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all prediction metrics."""
    metrics = {}
    metrics.update(compute_regression_metrics(y_true, y_pred))
    metrics.update(compute_directional_metrics(y_true, y_pred))
    return metrics
