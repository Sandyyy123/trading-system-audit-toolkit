"""Look-ahead bias and target-leakage detectors.

The single most common reason a "next-day price predictor" looks brilliant in a
notebook and dies in production: information from the future (or from the target
itself) leaks into the training features. These checks catch the three usual
culprits.
"""
from __future__ import annotations

import numpy as np


def detect_target_leakage(features: dict[str, np.ndarray],
                          target: np.ndarray,
                          corr_threshold: float = 0.95) -> list[dict]:
    """Flag any feature that leaks the target.

    Correlation is measured in *return space* (first differences), not on the
    price level. This matters: because daily prices are near-random-walks, every
    price-level feature (an SMA, yesterday's close, ...) correlates ~0.99 with
    the next price level. That is autocorrelation, not leakage. Detrending first
    means only a feature that actually contains the target's next-day *move*
    - i.e. the target in disguise - trips the detector.
    """
    tgt_ret = np.diff(np.asarray(target, dtype=float))
    findings = []
    for name, col in features.items():
        col = np.asarray(col, dtype=float)
        col_ret = np.diff(col)
        if col_ret.std() == 0 or tgt_ret.std() == 0:
            continue
        c = abs(np.corrcoef(col_ret, tgt_ret)[0, 1])
        if c >= corr_threshold:
            findings.append({
                "feature": name,
                "abs_corr_with_target_return": round(float(c), 4),
                "severity": "CRITICAL",
                "issue": "Feature's day-to-day move tracks the target's next-day "
                         "move almost perfectly - the target in disguise "
                         "(unshifted close or future-derived column).",
            })
    return findings


def check_feature_lag(feature_available_at: np.ndarray,
                      decision_time: np.ndarray) -> dict:
    """Every feature value must be knowable strictly *before* the decision bar.
    Flags rows where a feature timestamp is at or after the moment we act.
    """
    violations = int(np.sum(feature_available_at >= decision_time))
    return {
        "rows_checked": int(len(decision_time)),
        "lookahead_violations": violations,
        "severity": "CRITICAL" if violations else "OK",
        "issue": ("Feature timestamped at/after the decision bar - the model is "
                  "trained on information it would not have live.")
        if violations else "All features precede their decision bar.",
    }


def check_time_ordered_split(train_idx: np.ndarray, test_idx: np.ndarray) -> dict:
    """Time-series validation must never shuffle: every test index has to come
    after the last train index. Random k-fold on a price series leaks the future
    into the past and inflates every metric.
    """
    ok = train_idx.max() < test_idx.min()
    return {
        "train_end": int(train_idx.max()),
        "test_start": int(test_idx.min()),
        "severity": "OK" if ok else "CRITICAL",
        "issue": "Chronological split - no leakage across the boundary."
        if ok else "Test window overlaps/precedes train - shuffled CV leaks the future.",
    }
