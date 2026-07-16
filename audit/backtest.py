"""Leak-free walk-forward backtest harness.

Expanding-window (or rolling-window) evaluation: the model is only ever fit on
data strictly before the bar it predicts, then a simple long/short-on-direction
strategy is realised on the out-of-sample forecasts. This is the only number
that means anything - in-sample fit is decoration.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import metrics


def walk_forward(prices: np.ndarray,
                 model_fn: Callable[[np.ndarray], float],
                 min_train: int = 60,
                 rolling: int | None = None) -> dict:
    """Run an expanding/rolling walk-forward evaluation.

    model_fn receives the price history *up to and including* day t and returns
    a forecast for day t+1's close. It never sees the future.

    Returns out-of-sample predictions plus a realised strategy equity curve:
    go long the next day's return when the forecast is up, short when down.
    """
    preds, actual_next, prev = [], [], []
    strat_rets = []

    for t in range(min_train, len(prices) - 1):
        lo = 0 if rolling is None else max(0, t - rolling)
        history = prices[lo:t + 1]
        fcast = float(model_fn(history))

        today = prices[t]
        tomorrow = prices[t + 1]
        realised_ret = tomorrow / today - 1.0
        position = 1.0 if fcast > today else -1.0  # long if predicted up else short

        preds.append(fcast)
        actual_next.append(tomorrow)
        prev.append(today)
        strat_rets.append(position * realised_ret)

    preds = np.array(preds)
    actual_next = np.array(actual_next)
    prev = np.array(prev)
    strat_rets = np.array(strat_rets)

    return {
        "n_oos_days": int(len(preds)),
        "directional_accuracy": round(metrics.directional_accuracy(prev, actual_next, preds), 4),
        "mae": round(metrics.mae(actual_next, preds), 5),
        "strategy_sharpe": round(metrics.sharpe(strat_rets), 3),
        "strategy_max_drawdown": round(metrics.max_drawdown(strat_rets), 4),
        "strategy_total_return": round(float(np.prod(1 + strat_rets) - 1), 4),
        "_strat_rets": strat_rets,
    }
