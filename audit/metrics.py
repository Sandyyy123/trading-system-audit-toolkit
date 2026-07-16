"""Out-of-sample evaluation metrics for next-day equity prediction systems.

Every metric here is deliberately the *honest* version: directional accuracy is
measured on the predicted *change* (not the price level, which is trivially
autocorrelated), and Sharpe / drawdown are computed on a realised strategy
equity curve, not on fitted values.
"""
from __future__ import annotations

import numpy as np


def directional_accuracy(prev_price: np.ndarray,
                         actual_next: np.ndarray,
                         pred_next: np.ndarray) -> float:
    """Fraction of days where the model got the *direction* of the move right.

    A next-day-price model that only learns "tomorrow is about the same as
    today" will score ~50% here even though its price-level R^2 looks superb.
    """
    actual_dir = np.sign(actual_next - prev_price)
    pred_dir = np.sign(pred_next - prev_price)
    mask = actual_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float((actual_dir[mask] == pred_dir[mask]).mean())


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def information_coefficient(actual_ret: np.ndarray, pred_ret: np.ndarray) -> float:
    """Rank (Spearman) correlation between predicted and realised returns.

    IC ~ 0.03-0.05 is already a genuinely useful signal in daily equities.
    An in-sample IC of 0.9 is a red flag for leakage, not skill.
    """
    def _rank(x):
        order = x.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x))
        return ranks

    ra, rp = _rank(actual_ret), _rank(pred_ret)
    if ra.std() == 0 or rp.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rp)[0, 1])


def sharpe(strategy_returns: np.ndarray, periods_per_year: int = 252) -> float:
    r = np.asarray(strategy_returns, dtype=float)
    if r.std(ddof=1) == 0 or len(r) < 2:
        return float("nan")
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(strategy_returns: np.ndarray) -> float:
    """Worst peak-to-trough decline of the compounded equity curve (negative)."""
    equity = np.cumprod(1.0 + np.asarray(strategy_returns, dtype=float))
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return float(dd.min())
