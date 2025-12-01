"""
Trading performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_trading_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute trading performance metrics.

    Args:
        returns: Series of strategy returns

    Returns:
        Dictionary of performance metrics
    """
    # Basic return metrics
    total_return = (1 + returns).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0

    # Risk metrics
    volatility = returns.std() * np.sqrt(252)

    # Sharpe ratio
    sharpe = annual_return / (volatility + 1e-8)

    # Sortino ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
    sortino = annual_return / downside_std

    # Maximum drawdown
    equity_curve = (1 + returns).cumprod()
    rolling_max = equity_curve.expanding().max()
    drawdowns = equity_curve / rolling_max - 1
    max_drawdown = drawdowns.min()

    # Win rate
    win_rate = (returns > 0).mean()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate
    }
