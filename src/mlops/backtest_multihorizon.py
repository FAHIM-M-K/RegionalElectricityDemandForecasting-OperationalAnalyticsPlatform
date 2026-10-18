"""Small, reusable evaluation helpers for direct multi-horizon forecasts."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROMOTION_POLICY_VERSION = "rolling_win_rate_and_peak_guard_v1"
MIN_SELECTION_WINDOWS = 4
MIN_WINDOW_WIN_RATE = 0.75
MAX_PEAK_MAE_REGRESSION = 0.05


def evaluate_promotion_gate(window_metrics: list[Dict[str, float]]) -> Dict[str, Any]:
    """Require aggregate skill, repeatability, and bounded peak-hour regression.

    Each item contains candidate_mae, baseline_mae, candidate_peak_mae, and
    baseline_peak_mae for one chronological selection window. This rule is
    intentionally small and uses selection data only; the final holdout is not
    an input to promotion.
    """
    required = ("candidate_mae", "baseline_mae", "candidate_peak_mae", "baseline_peak_mae")
    if len(window_metrics) < MIN_SELECTION_WINDOWS:
        return {"accepted": False, "reason": f"requires at least {MIN_SELECTION_WINDOWS} selection windows",
                "window_count": len(window_metrics), "win_rate": np.nan,
                "candidate_mae": np.nan, "baseline_mae": np.nan,
                "candidate_peak_mae": np.nan, "baseline_peak_mae": np.nan}
    if any(any(key not in window or not np.isfinite(window[key]) for key in required)
           for window in window_metrics):
        return {"accepted": False, "reason": "selection window metrics are incomplete",
                "window_count": len(window_metrics), "win_rate": np.nan,
                "candidate_mae": np.nan, "baseline_mae": np.nan,
                "candidate_peak_mae": np.nan, "baseline_peak_mae": np.nan}

    candidate_mae = float(np.mean([w["candidate_mae"] for w in window_metrics]))
    baseline_mae = float(np.mean([w["baseline_mae"] for w in window_metrics]))
    candidate_peak = float(np.mean([w["candidate_peak_mae"] for w in window_metrics]))
    baseline_peak = float(np.mean([w["baseline_peak_mae"] for w in window_metrics]))
    win_rate = float(np.mean([w["candidate_mae"] < w["baseline_mae"] for w in window_metrics]))

    failures = []
    if candidate_mae >= baseline_mae:
        failures.append("aggregate MAE does not beat the seasonal baseline")
    if win_rate < MIN_WINDOW_WIN_RATE:
        failures.append(f"candidate wins only {win_rate:.0%} of selection windows")
    if candidate_peak > baseline_peak * (1.0 + MAX_PEAK_MAE_REGRESSION):
        failures.append(
            f"peak-hour MAE regresses by more than {MAX_PEAK_MAE_REGRESSION:.0%}"
        )
    return {"accepted": not failures, "reason": "; ".join(failures) if failures else "passed",
            "window_count": len(window_metrics), "win_rate": win_rate,
            "candidate_mae": candidate_mae, "baseline_mae": baseline_mae,
            "candidate_peak_mae": candidate_peak, "baseline_peak_mae": baseline_peak}


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


def compute_matched_metrics(y_true, forecasts: Dict[str, Any], peak_mask=None) -> Dict[str, Any]:
    """Score multiple forecasts on exactly the same valid target rows."""
    actual = np.asarray(y_true, dtype=float)
    predictions = {name: np.asarray(values, dtype=float) for name, values in forecasts.items()}
    if any(values.shape != actual.shape for values in predictions.values()):
        raise ValueError("Actuals and every forecast must have the same shape for matched scoring.")
    valid = np.isfinite(actual)
    for values in predictions.values():
        valid &= np.isfinite(values)
    base_valid = np.isfinite(actual)
    coverage_pct = 100.0 * valid.sum() / base_valid.sum() if base_valid.any() else 0.0
    peak = np.asarray(peak_mask, dtype=bool)[valid] if peak_mask is not None else None
    metrics = {
        name: compute_metrics(actual[valid], values[valid], peak)
        for name, values in predictions.items()
    }
    return {"sample_count": int(valid.sum()), "coverage_pct": round(float(coverage_pct), 2),
            "metrics": metrics}


def compute_lead_bucket_metrics(df: pd.DataFrame, forecasts: Dict[str, Any], peak_mask=None) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Score forecasts by operational lead range using the same target rows.

    H24 uses one 1–24 hour bucket. H168 separates day one, day two, and days
    three through seven. The buckets are diagnostic only and do not affect
    model promotion.
    """
    leads = pd.to_numeric(df["lead_hours"], errors="coerce").to_numpy(float)
    actual = pd.to_numeric(df["actual_demand_mw"], errors="coerce").to_numpy(float)
    peak = np.asarray(peak_mask, dtype=bool) if peak_mask is not None else None
    predictions = {name: np.asarray(values, dtype=float) for name, values in forecasts.items()}
    if any(values.shape != actual.shape for values in predictions.values()):
        raise ValueError("Lead-bucket forecasts must have the same shape as actuals.")
    horizon = int(df["horizon_hours"].iloc[0]) if len(df) else 0
    if horizon == 24:
        buckets = {"lead_1_24": (1, 24)}
    elif horizon == 168:
        buckets = {"lead_1_24": (1, 24), "lead_25_48": (25, 48), "lead_49_168": (49, 168)}
    else:
        buckets = {f"lead_1_{horizon}": (1, horizon)}

    result = {}
    for bucket, (lower, upper) in buckets.items():
        in_bucket = np.isfinite(leads) & (leads >= lower) & (leads <= upper)
        result[bucket] = {
            name: compute_metrics(actual[in_bucket], values[in_bucket], peak[in_bucket] if peak is not None else None)
            for name, values in predictions.items()
        }
    return result


def compute_threshold_metrics(y_true, y_pred, threshold_mw: float) -> Dict[str, float]:
    """Evaluate point-forecast threshold flags; this does not estimate risk probability."""
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actuals and predictions must have the same shape.")
    valid = np.isfinite(actual) & np.isfinite(predicted)
    actual_high = actual[valid] >= threshold_mw
    predicted_high = predicted[valid] >= threshold_mw
    tp = int(np.sum(actual_high & predicted_high))
    fp = int(np.sum(~actual_high & predicted_high))
    fn = int(np.sum(actual_high & ~predicted_high))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {"sample_count": int(valid.sum()), "actual_high_count": int(actual_high.sum()),
            "predicted_high_count": int(predicted_high.sum()), "true_positive": tp,
            "false_positive": fp, "false_negative": fn,
            "precision": round(precision, 4) if np.isfinite(precision) else np.nan,
            "recall": round(recall, 4) if np.isfinite(recall) else np.nan}


def compute_peak_event_metrics(df: pd.DataFrame, forecasts: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Measure peak height and peak-time error for complete forecast-origin paths."""
    required = {"forecast_origin_utc", "target_timestamp_utc", "horizon_hours", "lead_hours", "actual_demand_mw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Peak-event evaluation is missing columns: {sorted(missing)}")
    frame = df.reset_index(drop=True)
    actual = pd.to_numeric(frame["actual_demand_mw"], errors="coerce").to_numpy(float)
    predictions = {name: np.asarray(values, dtype=float) for name, values in forecasts.items()}
    if any(values.shape != actual.shape for values in predictions.values()):
        raise ValueError("Peak-event forecasts must have the same shape as actuals.")
    horizon = int(frame["horizon_hours"].iloc[0]) if len(frame) else 0
    expected_rows = horizon
    origin_values = pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    target_values = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    rows = []
    for origin, indexes in origin_values.groupby(origin_values).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        origin_leads = pd.to_numeric(frame.iloc[positions]["lead_hours"], errors="coerce")
        if (len(positions) != expected_rows or not np.isfinite(actual[positions]).all()
                or set(origin_leads) != set(range(1, horizon + 1))):
            continue
        if not np.all((target_values.iloc[positions] - origin).dt.total_seconds().to_numpy() ==
                      origin_leads.to_numpy(float) * 3600):
            continue
        actual_peak_pos = positions[int(np.argmax(actual[positions]))]
        origin_result = {"actual_peak_mw": float(actual[actual_peak_pos]),
                         "actual_peak_time": target_values.iloc[actual_peak_pos]}
        for name, values in predictions.items():
            if not np.isfinite(values[positions]).all():
                continue
            predicted_peak_pos = positions[int(np.argmax(values[positions]))]
            origin_result[name] = {
                "magnitude_error": abs(float(np.max(values[positions])) - float(np.max(actual[positions]))),
                "timing_error_hours": abs((target_values.iloc[predicted_peak_pos] - origin_result["actual_peak_time"]).total_seconds()) / 3600,
            }
        rows.append(origin_result)
    result = {}
    for name in predictions:
        values = [row[name] for row in rows if name in row]
        result[name] = {
            "complete_origins": len(values),
            "peak_magnitude_mae_mw": round(float(np.mean([item["magnitude_error"] for item in values])), 2) if values else np.nan,
            "peak_timing_mae_hours": round(float(np.mean([item["timing_error_hours"] for item in values])), 2) if values else np.nan,
        }
    return result


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
