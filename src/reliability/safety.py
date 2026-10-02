"""Availability guardrails for federated updates and forecast fallback decisions."""
from dataclasses import dataclass

import numpy as np


def update_norm(params, reference):
    return float(np.sqrt(sum(np.sum((p - r) ** 2, dtype=np.float64)
                             for p, r in zip(params, reference))))


def screen_updates(updates, reference, max_norm_ratio):
    """Reject non-finite or implausibly large updates using a robust peer-median threshold."""
    norms, rejected = {}, {}
    for update in updates:
        if hasattr(update, "n_samples") and update.n_samples < 1:
            rejected[update.name] = "insufficient_samples"
        elif hasattr(update, "e_global") and (not np.isfinite(update.e_global) or
                                             not np.isfinite(update.e_local)):
            rejected[update.name] = "invalid_validation_metric"
        elif hasattr(update, "quality") and (not np.isfinite(update.quality) or
                                             not 0 <= update.quality <= 1):
            rejected[update.name] = "invalid_quality_metric"
        elif len(update.params) != len(reference) or any(
                p.shape != r.shape for p, r in zip(update.params, reference)):
            rejected[update.name] = "invalid_update_shape"
        elif any(not np.isfinite(p).all() for p in update.params):
            rejected[update.name] = "non_finite_update"
        else:
            value = update_norm(update.params, reference)
            if not np.isfinite(value):
                rejected[update.name] = "update_norm_overflow"
            else:
                norms[update.name] = value
    positive = [v for v in norms.values() if v > 0]
    median = float(np.median(positive)) if positive else 0.0
    if median > 0:
        for name, norm in norms.items():
            if norm > max_norm_ratio * median:
                rejected[name] = "excessive_update_norm"
    accepted = [u for u in updates if u.name not in rejected]
    return accepted, rejected, norms


@dataclass(frozen=True)
class FallbackDecision:
    source: str
    reason: str


def choose_fallback(global_healthy, local_available=True):
    if global_healthy:
        return FallbackDecision("global_model", "healthy")
    if local_available:
        return FallbackDecision("local_model", "global_unhealthy")
    return FallbackDecision("smart_persistence", "global_and_local_unavailable")


def should_rollback(candidate_score, trusted_score, max_ratio):
    if not np.isfinite(candidate_score):
        return True, "non_finite_validation"
    if np.isfinite(trusted_score) and candidate_score > max_ratio * trusted_score:
        return True, "validation_regression"
    return False, ""


def validation_gate(reports, cfg):
    """Per-site absolute guardrails, frozen before experiments."""
    for report in reports:
        if not report["valid"]:
            return False, "invalid_predictions"
        rmse = np.sqrt(report["mse"])
        if rmse > cfg.get("max_site_validation_rmse", np.inf):
            return False, "worst_site_validation"
        if abs(report["bias"]) > cfg.get("max_validation_bias", np.inf):
            return False, "validation_bias"
        if rmse > max(cfg.get("baseline_rmse_floor", 0.10),
                      cfg.get("max_baseline_rmse_ratio", np.inf) * np.sqrt(report["baseline_mse"])):
            return False, "worse_than_smart_persistence"
    return True, ""


def safe_forecast(current, last_trusted, local, persistence, *, age_minutes=0,
                  max_age_minutes=30, stale_inputs=False, drift_alarm=False):
    """Choose a valid runtime forecast and record source, age and alert.

    Stale sensors force a persistence hold and a conservative reserve alert.
    The caller supplies checkpoint/local forecasts available at that site.
    """
    base = np.asarray(persistence, dtype=float)
    if not np.isfinite(base).all():
        raise ValueError("No valid persistence fallback is available")
    reason = "stale_inputs" if stale_inputs else "forecast_expired" if age_minutes > max_age_minutes \
        else "drift_alarm" if drift_alarm else ""
    candidates = [("current_trusted_model", current), ("last_trusted_model", last_trusted),
                  ("local_model", local)] if not reason else []
    for source, values in candidates:
        if values is not None:
            values = np.asarray(values, dtype=float)
            if values.shape == base.shape and np.isfinite(values).all():
                return np.clip(values, 0, 1), {"source": source,
                    "reason": "healthy" if source == "current_trusted_model" else "invalid_or_missing_model",
                    "forecast_age_minutes": age_minutes,
                    "operator_attention": source != "current_trusted_model"}
    return np.clip(base, 0, 1), {"source": "smart_persistence",
        "reason": reason or "invalid_or_missing_model", "forecast_age_minutes": age_minutes,
        "operator_attention": True}
