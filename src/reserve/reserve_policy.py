"""Causal uncertainty-margin policies for one aggregation node.

Forecast error convention:  e_t = actual_t - forecast_t   (MW)
A shortfall happens when e_t < -r_t (renewables came in below forecast by more than the reserve).

These functions calculate the *uncertainty margin*.  The scheduler separately adds the
expected demand-supply gap and enforces backup power / energy limits.

Policy A (fixed)     : m_t = fixed_fraction * reference_t
Policy B (n-sigma)   : m_t = max(alpha * sigma_e - mu_e, 0)
Policy C (empirical) : m_t = max(Q_(1-delta)(forecast-actual), 0)
    mu_e, sigma_e = rolling mean / std of PAST daytime errors. Only errors already observable
    when the forecast is issued are used (shifted by the forecast horizon) -> strictly causal.
    Rule inspired by Khaing, Kannan & Rao, "Accurate energy predictions for optimal power flow
    problems", Clean Energy 2026. If errors are not Gaussian, realised coverage can differ from
    1 - delta; we report the realised value rather than assume it.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def fixed_reserve(forecast_mw, fraction):
    return np.maximum(fraction * np.asarray(forecast_mw, float), 0.0)


def nsigma_reserve(node, delta, window, min_periods, horizon, fallback=None):
    """node: DataFrame sorted by time with columns block, daytime, actual_mw, forecast_mw.
    Returns (reserve_mw, mu_e, sigma_e) aligned with node rows (0 reserve at night)."""
    alpha = norm.ppf(1 - delta)
    mu = pd.Series(np.nan, index=node.index)
    sd = pd.Series(np.nan, index=node.index)
    for _, blk in node[node["daytime"]].groupby("block"):
        e = blk["actual_mw"] - blk["forecast_mw"]
        known = e.shift(horizon)                 # error of target t is only known h steps later
        mu[blk.index] = known.rolling(window, min_periods=min_periods).mean()
        sd[blk.index] = known.rolling(window, min_periods=min_periods).std()
    r = np.maximum(alpha * sd - mu, 0.0)
    if fallback is None:
        fallback = np.zeros(len(node), dtype=float)
    r = r.fillna(pd.Series(np.asarray(fallback, float), index=node.index))
    r[~node["daytime"]] = 0.0
    return r.to_numpy(), mu.to_numpy(), sd.to_numpy()


def empirical_reserve(node, delta, window, min_periods, horizon, fallback=None):
    """Causal lower-tail margin without a Gaussian shape assumption.

    Uses the rolling (1-delta) quantile of ``forecast - actual``.  ``method='higher'``
    chooses an observed order statistic rather than interpolating into the tail.  This is
    a practical benchmark, not a finite-sample guarantee for autocorrelated time series.
    """
    q = pd.Series(np.nan, index=node.index)
    for _, blk in node[node["daytime"]].groupby("block"):
        loss = blk["forecast_mw"] - blk["actual_mw"]
        known = loss.shift(horizon)
        q[blk.index] = known.rolling(window, min_periods=min_periods).apply(
            lambda x: np.quantile(x, 1 - delta, method="higher"), raw=True)
    margin = np.maximum(q, 0.0)
    if fallback is None:
        fallback = np.zeros(len(node), dtype=float)
    margin = margin.fillna(pd.Series(np.asarray(fallback, float), index=node.index))
    margin[~node["daytime"]] = 0.0
    return margin.to_numpy(), q.to_numpy()
