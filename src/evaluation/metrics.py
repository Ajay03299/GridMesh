"""Forecast metrics. All regression metrics are computed on DAYTIME targets only
(night-time PV is trivially zero and would flatter every model).
Units: p.u. of installed capacity, so MAE = 0.05 means 5% of capacity."""
import numpy as np


def regression_metrics(y_true, y_pred, mask=None):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    err = y_pred - y_true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "n": int(len(err)),
    }


def persistence_forecast(split):
    """Naive persistence: PV(t+h) = PV(t)."""
    return split.pv_now


def smart_persistence(pv_now, cs_now, cs_future):
    """Clear-sky persistence: keep the current 'cloudiness ratio', move the sun forward.
    PV(t+h) = PV(t) * CS(t+h) / CS(t). A standard, strong solar baseline."""
    ratio = np.where(cs_now > 20, cs_future / np.maximum(cs_now, 1), 0.0)
    return np.clip(pv_now * ratio, 0, 1).astype(np.float32)


def smart_persistence_forecast(split):
    return smart_persistence(split.pv_now, split.cs_now, split.cs_future)


def skill_score(model_rmse, reference_rmse):
    """1 - RMSE_model / RMSE_ref. > 0 means better than the reference."""
    return float(1 - model_rmse / reference_rmse)
