"""End-to-end demo audit of a next-day US-equity price predictor.

Run:  python main.py

The script fabricates a realistic daily price series and a *plausible-looking*
"model" that secretly leaks the future, then runs the full audit and prints the
findings a client would receive. It shows the three things this audit exists to
catch:

  1. Target leakage  -> an in-sample R^2 that looks brilliant but is fraudulent
  2. Shuffled CV     -> future-into-past leakage across the train/test boundary
  3. No real edge    -> once the leak is removed, the model cannot beat a random walk

Dependencies: numpy only.
"""
from __future__ import annotations

import numpy as np

from audit import backtest, baselines, leakage, metrics


def make_price_series(n: int = 750, seed: int = 11) -> np.ndarray:
    """Geometric random walk with a small positive drift - behaves like a real
    large-cap daily close (near-unpredictable day to day)."""
    rng = np.random.default_rng(seed)
    daily_ret = rng.normal(0.0003, 0.012, n)  # ~7.5% annual drift, ~19% annual vol
    return 100.0 * np.exp(np.cumsum(daily_ret))


def leaky_model_features(prices: np.ndarray) -> dict[str, np.ndarray]:
    """What a naive dev often builds: features aligned to the SAME row as the
    target. 'tomorrow_close_feature' is the target itself, un-shifted - the
    classic leak that produces a 0.99 R^2 in the notebook."""
    target_next_close = prices[1:]                 # what we are trying to predict
    return {
        "sma_5": np.convolve(prices, np.ones(5) / 5, "same")[:-1],
        "rsi_proxy": np.gradient(prices)[:-1],
        "tomorrow_close_feature": target_next_close,   # <-- LEAK (== target)
    }, target_next_close


def banner(txt: str) -> None:
    print("\n" + "=" * 68)
    print(txt)
    print("=" * 68)


def main() -> None:
    prices = make_price_series()
    banner("SYNTHETIC US-EQUITY DAILY CLOSE  (n=%d days)" % len(prices))
    print(f"start ${prices[0]:.2f}  ->  end ${prices[-1]:.2f}   "
          f"realised total return {prices[-1] / prices[0] - 1:+.1%}")

    # ------------------------------------------------------------------ #
    # FINDING 1 - target leakage
    # ------------------------------------------------------------------ #
    features, target = leaky_model_features(prices)
    leak_findings = leakage.detect_target_leakage(features, target)
    banner("FINDING 1 - TARGET LEAKAGE SCAN")
    if leak_findings:
        for f in leak_findings:
            print(f"  [{f['severity']}] {f['feature']}: "
                  f"|corr| with target return = {f['abs_corr_with_target_return']}")
            print(f"           {f['issue']}")
    else:
        print("  No leakage detected.")

    # ------------------------------------------------------------------ #
    # FINDING 2 - shuffled cross-validation
    # ------------------------------------------------------------------ #
    n = len(target)
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(n)
    bad_split = leakage.check_time_ordered_split(shuffled[: n // 2], shuffled[n // 2:])
    good_split = leakage.check_time_ordered_split(np.arange(n // 2), np.arange(n // 2, n))
    banner("FINDING 2 - TRAIN/TEST SPLIT INTEGRITY")
    print(f"  Shuffled k-fold : [{bad_split['severity']}] {bad_split['issue']}")
    print(f"  Chronological   : [{good_split['severity']}] {good_split['issue']}")

    # ------------------------------------------------------------------ #
    # FINDING 3 - does an honest model beat a random walk?
    # ------------------------------------------------------------------ #
    # An honest, non-leaky forecaster: trailing 20-day drift.
    honest_pred = baselines.drift_forecast(prices, lookback=20)
    cmp = baselines.compare_to_baseline(prices, honest_pred)
    banner("FINDING 3 - EDGE vs RANDOM WALK  (leak removed, out-of-sample)")
    print(f"  model         : MAE {cmp['model']['mae']:.4f}   "
          f"dir-acc {cmp['model']['dir_acc']:.1%}")
    print(f"  error bar     : random-walk MAE {cmp['random_walk_mae']:.4f}  "
          f"(model must be lower)")
    print(f"  direction bar : coin-flip {cmp['coin_flip_dir']:.0%}  "
          f"(model must be higher)")
    print(f"  -> {cmp['verdict']}")

    # ------------------------------------------------------------------ #
    # Leak-free walk-forward with realised strategy metrics
    # ------------------------------------------------------------------ #
    wf = backtest.walk_forward(
        prices,
        model_fn=lambda hist: baselines.drift_forecast(hist, 20)[-1],
        min_train=60,
    )
    banner("WALK-FORWARD BACKTEST  (realised long/short-on-direction strategy)")
    print(f"  OOS days           : {wf['n_oos_days']}")
    print(f"  directional acc    : {wf['directional_accuracy']:.1%}")
    print(f"  strategy Sharpe    : {wf['strategy_sharpe']}")
    print(f"  strategy max DD     : {wf['strategy_max_drawdown']:.1%}")
    print(f"  strategy total ret : {wf['strategy_total_return']:+.1%}")

    banner("AUDIT SUMMARY")
    print("  1. In-notebook R^2 was inflated by a leaked target column.")
    print("  2. Reported accuracy used shuffled CV (future leaks into the past).")
    print("  3. With the leak removed, the model does not beat a random walk.")
    print("  Recommendation: fix the split, drop leaked features, re-benchmark")
    print("  against random-walk + drift before any capital is committed.")


if __name__ == "__main__":
    main()
