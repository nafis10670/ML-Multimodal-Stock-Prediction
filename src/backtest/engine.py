"""
Backtesting engine for trading strategy evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    equity_curve: pd.Series


class BacktestEngine:
    """Backtesting engine for trading strategies."""

    def __init__(
        self,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        max_position: float = 1.0
    ):
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_position = max_position

    def run(
        self,
        predictions: pd.Series,
        prices: pd.Series,
        returns: pd.Series,
        signal_threshold: float = 0.0,
        vix: Optional[pd.Series] = None,
        vix_threshold: float = 30.0
    ) -> BacktestResult:
        """Run backtest with given predictions."""
        # Align data
        common_idx = predictions.index.intersection(
            prices.index
        ).intersection(returns.index)

        predictions = predictions.loc[common_idx]
        prices = prices.loc[common_idx]
        returns = returns.loc[common_idx]

        if vix is not None:
            vix = vix.reindex(common_idx).fillna(method='ffill')

        # Initialize tracking
        positions = pd.Series(index=common_idx, dtype=float)
        strategy_returns = pd.Series(index=common_idx, dtype=float)
        trades = []

        current_position = 0.0

        for date in common_idx:
            pred = predictions.loc[date]
            price = prices.loc[date]
            actual_return = returns.loc[date]

            # Risk filter
            if vix is not None and not pd.isna(vix.loc[date]) and vix.loc[date] > vix_threshold:
                target_position = 0.0
            # Generate signal
            elif pred > signal_threshold:
                target_position = min(pred / 0.01, self.max_position)
                target_position = np.clip(target_position, 0, self.max_position)
            elif pred < -signal_threshold:
                target_position = max(pred / 0.01, -self.max_position)
                target_position = np.clip(target_position, -self.max_position, 0)
            else:
                target_position = 0.0

            # Calculate trade
            trade_size = target_position - current_position

            # Calculate costs
            cost_bps = (
                abs(trade_size) *
                (self.transaction_cost_bps + self.slippage_bps) / 10000
            )

            # Calculate strategy return
            position_return = current_position * actual_return
            net_return = position_return - cost_bps

            # Log trade if position changed
            if abs(trade_size) > 0.01:
                trades.append({
                    'date': date,
                    'action': 'buy' if trade_size > 0 else 'sell',
                    'trade_size': trade_size,
                    'price': price,
                    'costs': cost_bps
                })

            # Update state
            positions.loc[date] = target_position
            strategy_returns.loc[date] = net_return
            current_position = target_position

        # Calculate equity curve
        equity_curve = (1 + strategy_returns).cumprod()

        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns, returns, trades)

        return BacktestResult(
            returns=strategy_returns,
            positions=positions,
            trades=pd.DataFrame(trades) if trades else pd.DataFrame(),
            metrics=metrics,
            equity_curve=equity_curve
        )

    def _calculate_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        trades: list
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1 if len(strategy_returns) > 0 else 0

        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / (volatility + 1e-8)

        downside_returns = strategy_returns[strategy_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-8
        sortino = annual_return / downside_std

        equity_curve = (1 + strategy_returns).cumprod()
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve / rolling_max - 1
        max_drawdown = drawdowns.min()

        win_rate = (strategy_returns > 0).mean()
        n_trades = len(trades)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': n_trades
        }
