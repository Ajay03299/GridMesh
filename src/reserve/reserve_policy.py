"""Reserve policies for one aggregation node.

Forecast error convention:  e_t = actual_t - forecast_t   (MW)
A shortfall happens when e_t < -r_t (renewables came in below forecast by more than the reserve).

Policy A (fixed)  : r_t = fixed_fraction * forecast_t
Policy B (n-sigma): r_t = max(alpha * sigma_e - mu_e, 0),  alpha = Phi^-1(1 - delta)
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


def nsigma_reserve(node, delta, window, min_periods, horizon):
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
    r = np.maximum(alpha * sd - mu, 0.0).fillna(0.0)
    r[~node["daytime"]] = 0.0
    return r.to_numpy(), mu.to_numpy(), sd.to_numpy()
