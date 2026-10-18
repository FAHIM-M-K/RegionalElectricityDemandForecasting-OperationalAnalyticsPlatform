"""Small, reusable evaluation helpers for direct multi-horizon forecasts."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def compute_metrics(y_true, y_pred, peak_mask=None) -> Dict[str, float]:
    actual, predicted = np.asarray(y_true, float), np.asarray(y_pred, float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {"mae": np.nan, "rmse": np.nan, "bias": np.nan, "smape": np.nan, "peak_mae": np.nan, "sample_count": 0}
    errors = predicted[valid] - actual[valid]
    absolute = np.abs(errors)
    denominator = (np.abs(actual[valid]) + np.abs(predicted[valid])) / 2
    smape = np.mean(absolute[denominator > 100] / denominator[denominator > 100]) * 100 if (denominator > 100).any() else np.nan
    peak_mae = np.nan
    if peak_mask is not None:
        peak = np.asarray(peak_mask, bool)[valid]
        if peak.any():
            peak_mae = float(absolute[peak].mean())
    return {"mae": round(float(absolute.mean()), 2), "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 2),
            "bias": round(float(errors.mean()), 2), "smape": round(float(smape), 2) if np.isfinite(smape) else np.nan,
            "peak_mae": round(peak_mae, 2) if np.isfinite(peak_mae) else np.nan, "sample_count": int(valid.sum())}


def compute_naive_baselines(df: pd.DataFrame, horizon: int | None = None):
    """Return exact T-24 and T-168 benchmarks precomputed by the example builder.

    These values are valid at the origin for their intended leads.  We do not
    reconstruct them from an origin snapshot because that was the former bug.
    """
    daily = pd.to_numeric(df.get("daily_naive_mw"), errors="coerce").to_numpy(float)
    weekly = pd.to_numeric(df.get("weekly_naive_mw"), errors="coerce").to_numpy(float)
    return daily, weekly


def generate_chronological_splits(origins: pd.Series, n_test_windows: int = 4, window_days: int = 30,
                                  holdout_days: int = 30, min_train_days: int = 365) -> List[Dict[str, Any]]:
    """Create rolling selection windows plus one untouched final holdout.

    Splits are based on forecast origins, so every prediction remains evaluated
    at the time it would actually have been issued.
    """
    unique = pd.Series(pd.to_datetime(origins, utc=True).unique()).sort_values().reset_index(drop=True)
    if unique.empty:
        raise ValueError("Cannot split an empty origin series.")
    final_start = unique.iloc[-1] - pd.Timedelta(days=holdout_days)
    selection_end = final_start
    splits: List[Dict[str, Any]] = []
    for index in range(n_test_windows, 0, -1):
        test_end = selection_end - pd.Timedelta(days=(index - 1) * window_days)
        test_start = test_end - pd.Timedelta(days=window_days)
        train_end = test_start
        train_start = unique.iloc[0]
        if train_end - train_start >= pd.Timedelta(days=min_train_days):
            splits.append({"kind": "selection", "train_start": train_start, "train_end": train_end,
                           "test_start": test_start, "test_end": test_end})
    if not splits:
        raise ValueError("Not enough history for a rolling backtest and final holdout.")
    splits.append({"kind": "final_holdout", "train_start": unique.iloc[0], "train_end": final_start,
                   "test_start": final_start, "test_end": unique.iloc[-1] + pd.Timedelta(hours=1)})
    return splits
