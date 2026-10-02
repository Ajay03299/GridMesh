"""Experiment runners and the common evaluation used by EVERY method.

Each runner returns a Result: per-client test forecasts + metrics, so all methods are
compared on exactly the same held-out data (daytime targets, p.u. of capacity).
"""
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd

from src.data.preprocessing import Split
from src.evaluation.metrics import (regression_metrics, persistence_forecast,
                                    smart_persistence_forecast)
from src.models.forecasting_model import build_model, fit_with_best_val, predict, get_params, n_bytes
from src.models.tree_models import (build_decision_tree, build_xgboost, feature_importance,
                                    fit_tree, model_bytes, timed_prediction)


@dataclass
class Result:
    method: str
    preds: dict                      # site name -> np.ndarray test forecast (p.u.)
    history: list = field(default_factory=list)   # training/round log
    val_preds: dict = field(default_factory=dict) # site name -> validation forecast (reserve warm-up)
    extra: dict = field(default_factory=dict)     # comms bytes, weights, etc.


def evaluate(result, clients):
    """Per-site + pooled metrics on daytime test targets."""
    per_site, ys, ps = {}, [], []
    for c in clients:
        m = c.test.daytime
        pred = result.preds[c.params.name]
        per_site[c.params.name] = regression_metrics(c.test.y, pred, m)
        ys.append(c.test.y[m]); ps.append(pred[m])
    pooled = regression_metrics(np.concatenate(ys), np.concatenate(ps))
    out = {
        "method": result.method,
        "global_mae": pooled["mae"], "global_rmse": pooled["rmse"], "global_bias": pooled["bias"],
        "worst_site_rmse": max(v["rmse"] for v in per_site.values()),
        "worst_site": max(per_site, key=lambda k: per_site[k]["rmse"]),
        "mean_site_mae": float(np.mean([v["mae"] for v in per_site.values()])),
    }
    # damage to HEALTHY sites when some sites are faulty (test data is always the clean data)
    faulty = result.extra.get("faulty_sites", [])
    healthy = [c for c in clients if c.params.name not in faulty]
    hy = np.concatenate([c.test.y[c.test.daytime] for c in healthy])
    hp = np.concatenate([result.preds[c.params.name][c.test.daytime] for c in healthy])
    hm = regression_metrics(hy, hp)
    out |= {"healthy_rmse": hm["rmse"], "healthy_mae": hm["mae"], "faulty_sites": ",".join(faulty)}
    # federated / system metrics
    rd = result.extra.get("rounds")
    if rd is not None:
        best = rd["val_mse_healthy"].min()
        out |= {
            "rounds": int(len(rd)),
            "rounds_to_converge": int(rd.loc[rd["val_mse_healthy"] <= 1.02 * best, "round"].iloc[0]),
            "participation_rate": float(rd["n_active"].mean() / len(clients)),
            "total_comm_mb": float(rd["bytes"].sum() / 1e6),
            "mean_comm_kb_per_round": float(rd["bytes"].mean() / 1e3),
            "rounds_skipped": int((~rd["aggregated"]).sum()),
            "selected_round": int(result.extra.get("best_round", len(rd))),
            "dropped_client_events": int(rd["n_dropped"].sum()),
        }
    out["per_site"] = per_site
    return out


# ------------------------------------------------------------------ baselines
def run_persistence(clients, smart=True):
    fn = smart_persistence_forecast if smart else persistence_forecast
    return Result("smart_persistence" if smart else "persistence",
                  {c.params.name: fn(c.test) for c in clients},
                  val_preds={c.params.name: fn(c.val) for c in clients})


def _site_data(c):
    """The data a site actually holds: its degraded copy in fault experiments (baselines see the
    fault for the whole training run). Evaluation always uses the clean test data."""
    return c.faulty if c.faulty is not None else c


def run_local_only(clients, cfg, seed):
    """One independent model per site, trained only on that site's data."""
    preds, vpreds, hist = {}, {}, []
    training = inference = 0.0
    sizes = 0
    for c in clients:
        d = _site_data(c)
        model = build_model(d.train.X.shape[1], cfg["model"], seed)
        started = time.perf_counter()
        model, h = fit_with_best_val(model, d.train, d.val, cfg["model"], seed)
        training += time.perf_counter() - started
        sizes += n_bytes(get_params(model))
        started = time.perf_counter()
        preds[c.params.name] = predict(model, c.test)
        vpreds[c.params.name] = predict(model, c.val)
        inference += time.perf_counter() - started
        hist += [{"site": c.params.name, **r} for r in h]
    return Result("local_only", preds, hist, val_preds=vpreds, extra={
        "training_seconds": training, "inference_seconds": inference, "model_bytes": sizes,
        "data_arrangement": "local only"})


def _pool(splits):
    return Split(**{f: np.concatenate([getattr(s, f) for s in splits])
                    for f in Split.__dataclass_fields__})


def run_centralized(clients, cfg, seed):
    """Upper-bound benchmark: pool every site's (already standardised) data in one place.
    Not the proposed architecture — raw data would have to leave the sites."""
    train = _pool([_site_data(c).train for c in clients])
    val = _pool([_site_data(c).val for c in clients])
    model = build_model(train.X.shape[1], cfg["model"], seed)
    # Virtual sites share one weather record, so the pool repeats every weather day once per
    # site and one epoch over 100 sites = 25 passes over the same days (over-training). Each
    # epoch therefore uses a random 4/N share of the rows: same budget and validation checks
    # as with 4 sites, identical results at 4 sites. Real multi-site data keeps full epochs.
    frac = 1.0 if cfg["data"].get("site_col") else min(1.0, 4 / len(clients))
    started = time.perf_counter()
    model, hist = fit_with_best_val(model, train, val, cfg["model"], seed, frac=frac)
    training = time.perf_counter() - started
    started = time.perf_counter()
    preds = {c.params.name: predict(model, c.test) for c in clients}
    vpreds = {c.params.name: predict(model, c.val) for c in clients}
    return Result("centralized", preds, hist, val_preds=vpreds, extra={
        "training_seconds": training, "inference_seconds": time.perf_counter() - started,
        "model_bytes": n_bytes(get_params(model)), "data_arrangement": "centralized pooled raw rows"})


def _tree_outputs(model, clients):
    preds, vpreds, inference = {}, {}, 0.0
    for c in clients:
        preds[c.params.name], elapsed = timed_prediction(model, c.test)
        vpreds[c.params.name], velapsed = timed_prediction(model, c.val)
        inference += elapsed + velapsed
    return preds, vpreds, inference


def run_decision_tree(clients, cfg, seed):
    """Transparent pooled tree baseline; raw-data pooling makes this an oracle, not GridMesh."""
    train = _pool([_site_data(c).train for c in clients])
    model, seconds = fit_tree(build_decision_tree(cfg, seed), train)
    preds, vpreds, inference = _tree_outputs(model, clients)
    return Result("decision_tree", preds, val_preds=vpreds, extra={
        "training_seconds": seconds, "inference_seconds": inference,
        "model_bytes": model_bytes(model), "data_arrangement": "centralized pooled raw rows",
        "feature_importance": feature_importance(model, clients[0].feature_names)})


def run_local_xgboost(clients, cfg, seed, quick=False):
    """Independent per-site XGBoost models: local rows, but no cross-site learning."""
    preds, vpreds, importance, sizes = {}, {}, [], []
    training = inference = 0.0
    for i, c in enumerate(clients):
        d = _site_data(c)
        model, elapsed = fit_tree(build_xgboost(cfg, seed + i, quick), d.train, d.val)
        training += elapsed
        preds[c.params.name], ptime = timed_prediction(model, c.test)
        vpreds[c.params.name], vtime = timed_prediction(model, c.val)
        inference += ptime + vtime
        sizes.append(model_bytes(model))
        importance.append(feature_importance(model, c.feature_names))
    keys = clients[0].feature_names
    mean_importance = {k: float(np.mean([v[k] for v in importance])) for k in keys}
    return Result("local_xgboost", preds, val_preds=vpreds, extra={
        "training_seconds": training, "inference_seconds": inference,
        "model_bytes": int(sum(sizes)), "data_arrangement": "local only; no raw-data transfer",
        "feature_importance": mean_importance})


def run_centralized_xgboost(clients, cfg, seed, quick=False):
    """Strong pooled-data oracle using the same features, residual target and splits."""
    train = _pool([_site_data(c).train for c in clients])
    val = _pool([_site_data(c).val for c in clients])
    model, seconds = fit_tree(build_xgboost(cfg, seed, quick), train, val)
    preds, vpreds, inference = _tree_outputs(model, clients)
    return Result("centralized_xgboost", preds, val_preds=vpreds, extra={
        "training_seconds": seconds, "inference_seconds": inference,
        "model_bytes": model_bytes(model), "data_arrangement": "centralized pooled raw rows",
        "feature_importance": feature_importance(model, clients[0].feature_names)})


def predictions_frame(result, clients):
    """Long-format DataFrame of val + test forecasts — input for the reserve simulator.
    Validation rows directly precede test rows inside each month, so the reserve
    simulator can warm up its rolling error statistics without touching test data."""
    rows = []
    for c in clients:
        name = c.params.name
        for split_name, split, pred in (("val", c.val, result.val_preds.get(name)),
                                        ("test", c.test, result.preds[name])):
            if pred is None:
                continue
            rows.append(pd.DataFrame({
                "time": split.time, "split": split_name, "site": name,
                "capacity_mw": c.params.capacity_mw,
                "actual_pu": split.y, "forecast_pu": pred, "daytime": split.daytime,
                "forecast_source": result.extra.get("forecast_health", {}).get(name, {}).get(
                    "source", result.method),
                "forecast_age_minutes": result.extra.get("forecast_health", {}).get(name, {}).get(
                    "forecast_age_minutes", 0),
                "operator_attention": result.extra.get("forecast_health", {}).get(name, {}).get(
                    "operator_attention", False),
            }))
    return pd.concat(rows)


def save_predictions(result, clients, path):
    predictions_frame(result, clients).to_csv(path, index=False)
