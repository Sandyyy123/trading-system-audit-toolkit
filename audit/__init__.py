"""trading-system-audit-toolkit: a small, honest audit harness for next-day
US-equity prediction systems.

Modules
-------
metrics   : out-of-sample directional accuracy, IC, Sharpe, max drawdown
leakage   : look-ahead bias + target-leakage detectors
baselines : random-walk / drift benchmarks the model must beat
backtest  : leak-free walk-forward evaluation
"""
from . import backtest, baselines, leakage, metrics  # noqa: F401

__all__ = ["metrics", "leakage", "baselines", "backtest"]
__version__ = "0.1.0"
