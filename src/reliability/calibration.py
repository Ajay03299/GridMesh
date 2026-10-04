"""Causal rolling reserve diagnostics and a predefined conservative response."""
import numpy as np
import pandas as pd


def monitor_margin(node, margin, delta, window, min_periods, horizon,
                   fallback, tolerance=0.03, step_minutes=10):
    """Use only residuals observable at issue time to monitor previously issued margins.

    Insufficient/stale calibration or poor coverage uses max(proposed, fixed margin).
    This heuristic gives no distribution-free coverage guarantee.
    """
    margin, fixed = np.asarray(margin, float), np.asarray(fallback, float)
    count = np.zeros(len(node), dtype=int)
    coverage, age = np.full(len(node), np.nan), np.full(len(node), np.nan)
    used = margin.copy()
    sources = np.full(len(node), "nominal", dtype=object)
    reasons = np.full(len(node), "", dtype=object)
    actual = node.actual_mw.to_numpy()
    forecast = node.forecast_mw.to_numpy()
    daytime = node.daytime.to_numpy()
    times = pd.to_datetime(node.time).to_numpy()
    for _, block in node.groupby("block"):
        past = []
        indices = block.index.to_numpy()
        for pos in range(len(block)):
            idx = indices[pos]
            observed_pos = pos - horizon
            if observed_pos >= 0:
                old_idx = indices[observed_pos]
                if daytime[old_idx] and np.isfinite(actual[old_idx]) and np.isfinite(forecast[old_idx]):
                    past.append((times[old_idx], forecast[old_idx] - actual[old_idx] > used[old_idx]))
            history = past[-window:]
            count[idx] = len(history)
            if history:
                coverage[idx] = 1 - np.mean([x[1] for x in history])
                age[idx] = max(0.0, float((times[idx] - history[-1][0]) /
                                         np.timedelta64(1, "m")) - step_minutes * horizon)
            if not daytime[idx]:
                used[idx] = 0
                continue
            reason = "insufficient_calibration" if len(history) < min_periods \
                else "stale_calibration" if age[idx] > 60 \
                else "coverage_below_target" if coverage[idx] < 1 - delta - tolerance else ""
            if reason:
                used[idx] = max(margin[idx], fixed[idx])
                sources[idx], reasons[idx] = "conservative_fixed_floor", reason
    return pd.DataFrame({"guarded_margin_mw": used, "rolling_coverage": coverage,
                         "calibration_samples": count, "calibration_age_minutes": age,
                         "margin_source": sources, "calibration_warning": reasons}, index=node.index)
