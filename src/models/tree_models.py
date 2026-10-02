"""Leakage-safe tree forecasters for the tabular GridMesh feature matrix.

Trees learn the residual to smart persistence, just like the neural models.  Validation
data is used only for XGBoost early stopping; test data is never supplied to ``fit``.
"""
import pickle
import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor


def build_decision_tree(cfg, seed):
    tc = cfg["tree_models"]["decision_tree"]
    return DecisionTreeRegressor(random_state=seed, **tc)


def build_xgboost(cfg, seed, quick=False):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - clearer user-facing failure
        raise RuntimeError("XGBoost is required: pip install -r requirements.txt") from exc
    xc = dict(cfg["tree_models"]["xgboost"])
    if quick:
        xc["n_estimators"] = min(250, xc["n_estimators"])
        xc["early_stopping_rounds"] = min(25, xc["early_stopping_rounds"])
    return XGBRegressor(
        objective="reg:squarederror", random_state=seed, tree_method="hist",
        eval_metric="rmse", importance_type="gain", **xc)


def fit_tree(model, train, val=None):
    """Fit on training residuals; optionally use the time-ordered validation split."""
    kwargs = {}
    if val is not None and model.__class__.__name__.startswith("XGB"):
        kwargs["eval_set"] = [(val.X, val.y_model)]
        kwargs["verbose"] = False
    started = time.perf_counter()
    model.fit(train.X, train.y_model, **kwargs)
    return model, time.perf_counter() - started


def predict_tree(model, split):
    raw = split.base + model.predict(split.X)
    if not np.isfinite(raw).all():
        raise ValueError("Tree forecast contains invalid values; operator fallback required")
    return np.clip(raw, 0.0, 1.0).astype(np.float32)


def model_bytes(model):
    return len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))


def feature_importance(model, feature_names):
    values = getattr(model, "feature_importances_", np.zeros(len(feature_names)))
    return {name: float(value) for name, value in zip(feature_names, values)}


def feature_group(name):
    if name.startswith("pv_lag"):
        return "Recent PV"
    if name.startswith(("GHI", "DNI", "DHI", "kt_")):
        return "Irradiance"
    if name.startswith("Clearsky"):
        return "Clear-sky information"
    if "Temperature" in name or "Dew Point" in name:
        return "Temperature"
    if "Humidity" in name:
        return "Humidity"
    if name.startswith("cloud_"):
        return "Cloud type"
    if name.startswith(("cos_zenith", "hour_", "doy_")):
        return "Solar position / calendar"
    return "Other atmosphere"


def timed_prediction(model, split):
    started = time.perf_counter()
    pred = predict_tree(model, split)
    return pred, time.perf_counter() - started
