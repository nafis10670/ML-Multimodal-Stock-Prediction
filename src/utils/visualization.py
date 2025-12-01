"""
Plotting utilities.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve"):
    """Plot equity curve."""
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(True)
    return plt.gcf()


def plot_returns_distribution(returns: pd.Series, title: str = "Returns Distribution"):
    """Plot returns distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(returns, bins=50, edgecolor='black')
    axes[0].set_title(f"{title} - Histogram")
    axes[0].set_xlabel("Return")
    axes[0].set_ylabel("Frequency")

    axes[1].plot(returns.index, returns.values)
    axes[1].set_title(f"{title} - Time Series")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Return")
    axes[1].grid(True)

    plt.tight_layout()
    return fig


def plot_feature_importance(importance_dict: dict, top_n: int = 20):
    """Plot feature importance."""
    sorted_features = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    features, importances = zip(*sorted_features)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), importances)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Features")
    plt.tight_layout()
    return plt.gcf()
