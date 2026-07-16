"""Naive baselines every next-day price model must beat to be worth deploying.

The bar is not "positive R^2". Daily closes are ~random walks, so predicting
"tomorrow's close = today's close" already gives a huge price-level R^2. If the
model cannot beat that on *directional* accuracy and error, it has learned
nothing tradeable.
"""
from __future__ import annotations

import numpy as np

from . import metrics


def random_walk_forecast(prices: np.ndarray) -> np.ndarray:
    """Predict next close = current close. The honest benchmark."""
    return prices[:-1].copy()


def drift_forecast(prices: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Predict next close using the trailing mean daily drift - a slightly
    stronger baseline than pure random walk.
    """
    preds = np.empty(len(prices) - 1)
    log_ret = np.diff(np.log(prices))
    for t in range(len(prices) - 1):
        lo = max(0, t - lookback)
        mu = log_ret[lo:t].mean() if t > lo else 0.0
        preds[t] = prices[t] * np.exp(mu)
    return preds


def compare_to_baseline(prices: np.ndarray, model_pred_next: np.ndarray) -> dict:
    """Head-to-head: model vs random-walk on the same out-of-sample window.

    prices[t] is today, prices[t+1] is the actual next close, model_pred_next[t]
    is the model's forecast for prices[t+1].
    """
    prev = prices[:-1]
    actual_next = prices[1:]
    rw = random_walk_forecast(prices)

    model_mae = metrics.mae(actual_next, model_pred_next)
    model_dir = metrics.directional_accuracy(prev, actual_next, model_pred_next)
    rw_mae = metrics.mae(actual_next, rw)

    # Two honest bars: random walk sets the ERROR bar (it makes no directional
    # call, so 50% coin-flip - not the random walk - is the DIRECTION bar).
    beats_mae = model_mae < rw_mae
    beats_dir = model_dir > 0.5 + 1e-9
    return {
        "model": {"mae": round(model_mae, 5), "dir_acc": round(model_dir, 5)},
        "random_walk_mae": round(rw_mae, 5),
        "coin_flip_dir": 0.5,
        "beats_random_walk_on_error": bool(beats_mae),
        "beats_coin_flip_on_direction": bool(beats_dir),
        "verdict": ("Adds tradeable signal" if (beats_mae and beats_dir)
                    else "No edge over random-walk error / coin-flip direction - do not deploy"),
    }
