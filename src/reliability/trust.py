"""Client trust / reliability score (our prototype mechanism — NOT a published formula).

Each round, every reporting client k evaluates the CURRENT GLOBAL model on its own recent
validation window and reports that error e_k. Because every client scores the same model,
differences in e_k reflect the client's DATA (degraded sensors, bad labels), not its training.

    rel_k      = e_k / median_j(e_j)                          # error relative to peers
    r_k        = exp(-beta * max(0, rel_k - 1 - tol))         # 1 while within tolerance
    trust_k(t) = lam * trust_k(t-1) + (1 - lam) * r_k         # smoothed ("recent") reliability

A client with trust below `quarantine_below` is excluded from aggregation for that round
(it keeps reporting, so it can recover if its data becomes healthy again).
"""
import numpy as np


def instant_reliability(errors, tol, beta):
    """errors: dict name -> e_k. Returns dict name -> r_k in (0, 1]."""
    med = float(np.median(list(errors.values())))
    med = max(med, 1e-12)
    return {k: float(np.exp(-beta * max(0.0, e / med - 1.0 - tol))) for k, e in errors.items()}


def update_trust(trust, r_now, lam):
    """EMA update; clients not reporting this round keep their previous trust."""
    new = dict(trust)
    for k, r in r_now.items():
        new[k] = lam * trust.get(k, 1.0) + (1 - lam) * r
    return new


def data_quality(split, day_only=True):
    """Fresh-reading rate: share of (daytime) steps where the raw GHI sensor value changed
    since the previous step. Missing readings (forward-filled) and frozen sensors both lower it."""
    m = split.daytime if day_only else np.ones(len(split.stale), bool)
    return float(1.0 - split.stale[m].mean()) if m.any() else 1.0
