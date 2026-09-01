> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# trading-system-audit-toolkit

A small, honest audit harness for **next-day US-equity price prediction systems** —
the kind of automated swing-trading platform that looks brilliant in a notebook
and quietly loses money live.

It exists to answer one question a backtest chart never will:
**does this system have a real, tradeable edge, or is it overfit / leaking the future?**

---

## What it checks

| Module | Catches |
|--------|---------|
| `audit/leakage.py`   | Target leakage (un-shifted close as a "feature"), look-ahead bias (features timestamped after the decision bar), shuffled cross-validation on a time series |
| `audit/baselines.py` | Whether the model beats a **random walk** ("tomorrow ≈ today") and a trailing-drift benchmark — the real bar, not "positive R²" |
| `audit/backtest.py`  | Leak-free **walk-forward** evaluation with a realised long/short strategy equity curve |
| `audit/metrics.py`   | Directional accuracy on the *change* (not the level), information coefficient, Sharpe, max drawdown |

## Why "next-day price" systems fail

A daily close is close to a random walk, so predicting *the price level*
("tomorrow's close ≈ today's close") gives a huge R². Teams see that R² and
believe they have a predictor. They don't — until the model beats a random walk
on **direction** and **error**, out-of-sample, on a chronological split, there is
no edge. This toolkit measures exactly that.

## Run

```bash
pip install -r requirements.txt
python main.py
```

The demo builds a realistic synthetic price series and a deliberately *leaky*
model, then prints the findings a client receives: the leak that inflates the
in-notebook score, the shuffled-CV violation, and the collapse to no-edge once
the leak is removed.

## Audit workflow this maps to

1. **Reproduce** the reported metric — is it in-sample or out-of-sample?
2. **Leakage scan** — features vs target, feature timestamps, split integrity.
3. **Baseline gate** — beat random-walk + drift on direction and error, or stop.
4. **Walk-forward** — realised Sharpe, hit rate, max drawdown after costs.
5. **Regime check** — does the edge survive across volatility regimes, or only in the calm stretch it was fit on?

---

*Illustrative demo repository. Built to accompany a trading-system audit proposal.*
