"""Experiment runners and the common evaluation used by EVERY method.

Each runner returns a Result: per-client test forecasts + metrics, so all methods are
compared on exactly the same held-out data (daytime targets, p.u. of capacity).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.preprocessing import Split
from src.evaluation.metrics import (regression_metrics, persistence_forecast,
                                    smart_persistence_forecast)
from src.models.forecasting_model import build_model, fit_with_best_val, predict


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
    # damage to HEALTHY sites when one site is faulty (test data is always the clean data)
    faulty = result.extra.get("faulty_site")
    healthy = [c for c in clients if c.params.name != faulty]
    hy = np.concatenate([c.test.y[c.test.daytime] for c in healthy])
    hp = np.concatenate([result.preds[c.params.name][c.test.daytime] for c in healthy])
    hm = regression_metrics(hy, hp)
    out |= {"healthy_rmse": hm["rmse"], "healthy_mae": hm["mae"], "faulty_site": faulty}
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


def run_local_only(clients, cfg, seed):
    """One independent model per site, trained only on that site's data."""
    preds, vpreds, hist = {}, {}, []
    for c in clients:
        model = build_model(c.train.X.shape[1], cfg["model"], seed)
        model, h = fit_with_best_val(model, c.train, c.val, cfg["model"], seed)
        preds[c.params.name] = predict(model, c.test)
        vpreds[c.params.name] = predict(model, c.val)
        hist += [{"site": c.params.name, **r} for r in h]
    return Result("local_only", preds, hist, val_preds=vpreds)


def _pool(splits):
    return Split(**{f: np.concatenate([getattr(s, f) for s in splits])
                    for f in Split.__dataclass_fields__})


def run_centralized(clients, cfg, seed):
    """Upper-bound benchmark: pool every site's (already standardised) data in one place.
    Not the proposed architecture — raw data would have to leave the sites."""
    train, val = _pool([c.train for c in clients]), _pool([c.val for c in clients])
    model = build_model(train.X.shape[1], cfg["model"], seed)
    model, hist = fit_with_best_val(model, train, val, cfg["model"], seed)
    return Result("centralized", {c.params.name: predict(model, c.test) for c in clients}, hist,
                  val_preds={c.params.name: predict(model, c.val) for c in clients})


def save_predictions(result, clients, path):
    """Long-format CSV of val + test forecasts — input for the reserve simulator.
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
            }))
    pd.concat(rows).to_csv(path, index=False)
