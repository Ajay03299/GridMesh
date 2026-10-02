"""Automated correctness audit — run before every push:   python audit.py

Checks data integrity, leakage, the maths of every component, reserve causality and
calibration, fault injection, determinism and the real-data path. Exit code 1 if any fail.
"""
import copy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from run_simulation import run_method
from src.data.adapter import load_config, load_dataset
from src.data.preprocessing import (add_derived_features, make_windows, prepare_clients,
                                    segment_labels)
from src.data.pv_model import pv_power_pu
from src.data.virtual_sites import (apply_fault, generate_virtual_sites, measured_pu,
                                    sample_site_params)
from src.evaluation.experiments import _site_data
from src.evaluation.metrics import smart_persistence
from src.federated.client_selection import select_clients
from src.federated.fedavg import weighted_average
from src.federated.reliability_aware import ReliabilityAwareFedAvg
from src.federated.server import METADATA_BYTES, MODEL_TRANSFERS, run_federated
from src.reliability.drift import DriftDetector
from src.reserve.reserve_policy import empirical_reserve, nsigma_reserve
from src.reserve.simulator import schedule_backup, simulate
from src.models.forecasting_model import build_model
from src.models.tree_models import build_decision_tree, build_xgboost, fit_tree, predict_tree
from src.reliability.safety import (choose_fallback, screen_updates, should_rollback,
                                    safe_forecast, validation_gate)
from src.reliability.calibration import monitor_margin
from src.federated.hierarchical import HierarchicalReliabilityFedAvg

RESULTS = []


def check(name):
    def wrap(fn):
        t = time.time()
        try:
            detail = fn() or ""
            RESULTS.append((True, name, detail))
            print(f"  PASS  {name}  {detail}  ({time.time() - t:.1f}s)")
        except Exception as e:  # noqa: BLE001
            RESULTS.append((False, name, repr(e)))
            print(f"  FAIL  {name}  -> {repr(e)}")
        return fn
    return wrap


CFG = load_config()
SEED = CFG["seed"]
print("GridMesh audit")
BASE, META = load_dataset(CFG)
CLIENTS, INFO = prepare_clients(CFG, 4, SEED)


@check("data: one year, regular 10-min steps, no gaps or missing values")
def _():
    step = BASE["timestamp"].diff().dropna()
    assert (step == pd.Timedelta(minutes=10)).all()
    used = ["GHI", "DNI", "DHI", "Clearsky GHI", "Temperature", "Solar Zenith Angle", "Cloud Type"]
    assert BASE[used].isna().sum().sum() == 0
    return f"{len(BASE):,} rows"


@check("pv model: output in [0, 1], zero at night, rises with irradiance")
def _():
    pv = pv_power_pu(BASE["GHI"], BASE["Temperature"], 0.85, -0.004, CFG["pv_model"])
    assert pv.min() >= 0 and pv.max() <= 1
    assert (pv[BASE["GHI"].to_numpy() == 0] == 0).all()
    ramp = pv_power_pu(np.array([100, 400, 800]), np.array([25, 25, 25]), 0.85, -0.004,
                       CFG["pv_model"])
    assert np.all(np.diff(ramp) > 0)


@check("virtual sites: site k is identical whether you simulate 4 or 100 sites")
def _():
    a = sample_site_params(0, CFG["virtual_sites"], SEED)
    b = generate_virtual_sites(BASE.head(500), CFG, n_clients=6, seed=SEED)[0][0]
    assert vars(a) == vars(b)


@check("leakage: train / val / test target times never overlap")
def _():
    for c in CLIENTS:
        tr, va, te = (set(pd.to_datetime(s.time)) for s in (c.train, c.val, c.test))
        assert not (tr & va) and not (tr & te) and not (va & te)


@check("leakage: no input window crosses a train/val/test boundary")
def _():
    sdf = generate_virtual_sites(BASE, CFG, 1, SEED)[0][1]
    sdf["seg"] = segment_labels(sdf["timestamp"], CFG["split"])
    X, aux = make_windows(add_derived_features(sdf, CFG["features"]), CFG["features"])
    assert (aux["seg_in"] == aux["seg_out"]).all()
    return f"{len(X):,} windows"


@check("leakage: only deterministic values (sun position, clear-sky, calendar) from t+h")
def _():
    allowed = {"Clearsky GHI", "cos_zenith", "hour_sin", "hour_cos", "doy_sin", "doy_cos"}
    assert set(CFG["features"]["future_known"]) <= allowed
    future = [f for f in CLIENTS[0].feature_names if f.endswith("_t+h")]
    assert {f[:-4] for f in future} <= allowed


@check("leakage: scaling fitted on each site's TRAIN data only")
def _():
    for c in CLIENTS:
        X = c.train.X
        sd = X.std(axis=0)
        live = sd > 1e-3
        assert np.abs(X.mean(axis=0)[live]).max() < 0.05
        assert np.abs(sd[live] - 1).max() < 0.05
        assert np.abs(c.test.X.mean(axis=0)[live]).max() > 0.01   # test is NOT re-centred


@check("split: every site is validated and tested on all 12 months")
def _():
    for c in CLIENTS:
        for s in ("val", "test"):
            assert pd.to_datetime(getattr(c, s).time).month.nunique() == 12, (c.params.name, s)
    seen = {c.params.name: pd.to_datetime(c.train.time).month.nunique() for c in CLIENTS}
    assert min(seen.values()) >= 6
    return f"training months per site (younger plants see fewer seasons = non-IID): {seen}"


@check("smart persistence: exact when output tracks clear-sky")
def _():
    cs_now, cs_fut = np.array([200., 500., 800.]), np.array([300., 600., 700.])
    pred = smart_persistence(0.0008 * cs_now, cs_now, cs_fut)
    assert np.allclose(pred, 0.0008 * cs_fut, atol=1e-6)


@check("models: MLP, GRU and LSTM accept the same feature vector")
def _():
    n_features = CLIENTS[0].train.X.shape[1]
    for architecture in ("mlp", "gru", "lstm"):
        mcfg = copy.deepcopy(CFG["model"])
        mcfg["architecture"] = architecture
        model = build_model(n_features, mcfg, SEED)
        out = model(__import__("torch").from_numpy(CLIENTS[0].train.X[:8]))
        assert tuple(out.shape) == (8,) and np.isfinite(out.detach().numpy()).all()


@check("trees: decision tree fits residual target and returns bounded forecasts")
def _():
    c = CLIENTS[0]
    model, _ = fit_tree(build_decision_tree(CFG, SEED), c.train, c.val)
    pred = predict_tree(model, c.test)
    assert len(pred) == len(c.test.y) and np.isfinite(pred).all()
    assert pred.min() >= 0 and pred.max() <= 1


@check("trees: residual target is exactly actual minus smart persistence")
def _():
    c = CLIENTS[0]
    assert np.allclose(c.train.y_model, c.train.y - c.train.base)


@check("XGBoost: deterministic residual forecasts, validation-only early stopping, bounded output")
def _():
    from dataclasses import replace
    c = CLIENTS[0]
    cfg = copy.deepcopy(CFG)
    cfg["tree_models"]["xgboost"].update(n_estimators=20, early_stopping_rounds=5)
    models = [fit_tree(build_xgboost(cfg, SEED), c.train, c.val)[0] for _ in range(2)]
    a, b = [predict_tree(m, c.test) for m in models]
    assert np.array_equal(a, b) and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
    assert np.allclose(a, np.clip(c.test.base + models[0].predict(c.test.X), 0, 1))
    assert set(models[0].evals_result()) == {"validation_0"}
    assert len(models[0].evals_result()["validation_0"]["rmse"]) <= 20
    assert build_decision_tree(CFG, SEED).max_depth == 8
    # Changing held-out targets cannot alter the fitted model or predictions.
    changed = replace(c.test, y=np.ones_like(c.test.y))
    assert np.array_equal(a, predict_tree(models[0], changed))


@check("FedAvg: aggregation is the exact weighted average")
def _():
    a, b = [np.ones((2, 2)), np.zeros(3)], [np.full((2, 2), 5.0), np.ones(3)]
    out = weighted_average([a, b], [1, 3])
    assert np.allclose(out[0], 4.0) and np.allclose(out[1], 0.75)


@check("reliability weights: sum to 1, equal sites -> n*quality, outlier down-weighted, then quarantined")
def _():
    class U:
        def __init__(s, name, n, e, q):
            s.name, s.n_samples, s.e_global, s.quality = name, n, e, q
    agg = ReliabilityAwareFedAvg(CFG["reliability"])
    w, _ = agg.weights([U("A", 100, 1.0, 1.0), U("B", 300, 1.0, 1.0)])
    assert abs(sum(w.values()) - 1) < 1e-9 and abs(w["B"] - 0.75) < 1e-9
    agg = ReliabilityAwareFedAvg(CFG["reliability"])
    seen = []
    for _ in range(6):
        w, info = agg.weights([U("A", 100, 1.0, 1), U("B", 100, 1.0, 1), U("C", 100, 5.0, 1)])
        seen.append(w["C"])
    assert seen[0] < 1 / 3 and seen[-1] == 0.0 and "C" in info["quarantined"]
    return f"outlier weight {seen[0]:.3f} -> {seen[-1]:.3f}"


@check("drift detector: quiet on small wobbles, fires on a real jump")
def _():
    ecfg = CFG["event_aware"]
    d = DriftDetector(ecfg["drift_k"], ecfg["window"], ecfg["min_history"],
                      ecfg["min_rel_increase"])
    flags = [d.update("A", e) for e in [1.0, 1.01, 0.99, 1.0, 1.02, 1.03, 3.0]]
    assert flags[:-1] == [False] * 6 and flags[-1] is True


@check("client selection: sampling size, dropout, drifting sites always join")
def _():
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(10)]
    fcfg = {"client_fraction": 0.5, "dropout_rate": 0.0}
    act, _, drop = select_clients(names, rng, fcfg)
    assert len(act) == 5 and not drop
    ecfg = {"stable_participation": 0.0}
    act, _, _ = select_clients(names, rng, {"client_fraction": 1.0, "dropout_rate": 0.0},
                               ecfg, drifting={"S3"}, round_idx=2)
    assert act == ["S3"]


@check("client selection: max-clients cap bounds each round")
def _():
    names = [f"S{i}" for i in range(100)]
    act, _, _ = select_clients(names, np.random.default_rng(1),
                               {"client_fraction": 1.0, "dropout_rate": 0.0,
                                "max_clients_per_round": 12})
    assert len(act) == 12


@check("safety: non-finite and extreme-norm updates are rejected with reasons")
def _():
    class U:
        def __init__(s, name, value):
            s.name, s.params = name, [np.array([value], dtype=float)]
    accepted, rejected, _ = screen_updates(
        [U("normal", 1), U("peer", 1.1), U("extreme", 100), U("nan", np.nan)],
        [np.array([0.0])], 8.0)
    assert {u.name for u in accepted} == {"normal", "peer"}
    assert rejected == {"nan": "non_finite_update", "extreme": "excessive_update_norm"}


@check("safety: deterministic fallback chain and validation rollback gate")
def _():
    assert choose_fallback(False, True).source == "local_model"
    assert choose_fallback(False, False).source == "smart_persistence"
    assert should_rollback(np.nan, 1.0, 1.3)[0]
    assert should_rollback(1.5, 1.0, 1.3)[0]
    assert not should_rollback(1.1, 1.0, 1.3)[0]


@check("safety: real forecast fallback ordering, stale age and explicit failure")
def _():
    good, bad, base = np.array([0.4, 0.5]), np.array([np.nan, 0.5]), np.array([0.3, 0.4])
    for last, local, expected in [(good, good, "last_trusted_model"),
                                  (None, good, "local_model"),
                                  (None, None, "smart_persistence")]:
        p, record = safe_forecast(bad, last, local, base)
        assert np.isfinite(p).all() and record["source"] == expected
    _, record = safe_forecast(good, good, good, base, age_minutes=60)
    assert record["source"] == "smart_persistence" and record["operator_attention"]
    try:
        safe_forecast(None, None, None, bad)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid fallback must fail explicitly")


@check("rollback: prediction, worst-site, bias and persistence gates give explicit reasons")
def _():
    healthy = {"valid": True, "mse": 0.001, "bias": 0.0, "baseline_mse": 0.001}
    assert validation_gate([healthy], CFG["safety"])[0]
    for change, reason in [({"valid": False}, "invalid_predictions"),
                          ({"mse": 0.25}, "worst_site_validation"),
                          ({"bias": 0.2}, "validation_bias"),
                          ({"mse": 0.04}, "worse_than_smart_persistence")]:
        assert validation_gate([{**healthy, **change}], CFG["safety"]) == (False, reason)


@check("selection: quarantine exclusion, recovery probes and bounded drift priority")
def _():
    names = [f"S{i}" for i in range(20)]
    fc = {"client_fraction": 1, "dropout_rate": 0, "max_clients_per_round": 5}
    ec = {"stable_participation": 1}
    rng = np.random.default_rng(0)
    active, _, _ = select_clients(names, rng, fc, ec, drifting={"S19"}, round_idx=2,
                                  quarantined={"S0"})
    assert "S19" in active and "S0" not in active and len(active) == 5
    fc["max_clients_per_round"] = 20
    active, _, _ = select_clients(names, rng, fc, ec, round_idx=3, quarantined={"S0"})
    assert "S0" in active


@check("trust: quarantine recovers after healthy reports and all-quarantined weights stay zero")
def _():
    from types import SimpleNamespace
    agg = ReliabilityAwareFedAvg(CFG["reliability"])
    def us(error):
        return [SimpleNamespace(name=n, n_samples=100, quality=1, e_global=e)
                for n, e in [("A", 1), ("B", 1), ("C", error)]]
    for _ in range(6):
        weights, info = agg.weights(us(10))
    assert "C" in info["quarantined"]
    for _ in range(6):
        weights, info = agg.weights(us(1))
    assert weights["C"] > 0 and "C" not in info["quarantined"]
    agg.trust = {"A": 0, "B": 0, "C": 0}
    agg.rcfg = {**agg.rcfg, "trust_ema": 1.0}
    weights, _ = agg.weights(us(1))
    assert sum(weights.values()) == 0


@check("hierarchy: composed weights equal two-stage parameter averaging")
def _():
    from types import SimpleNamespace
    updates = [SimpleNamespace(name=n, n_samples=k, e_global=1.0,
                               quality=1.0, params=[np.array([v])])
               for n, k, v in [("A", 100, 1), ("B", 300, 5), ("C", 200, 7), ("D", 200, 9)]]
    agg = HierarchicalReliabilityFedAvg(CFG["reliability"], [u.name for u in updates], 2)
    weights, info = agg.weights(updates)
    assert abs(sum(weights.values()) - 1) < 1e-9
    combined = weighted_average([u.params for u in updates], [weights[u.name] for u in updates])
    assert np.allclose(combined[0], [6.0])
    assert info["reporting_groups"] == 2


SMALL = copy.deepcopy(CFG)
SMALL["federated"]["rounds"] = 3


@check("federated: deterministic (same seed -> identical forecasts)")
def _():
    r1 = run_federated(CLIENTS, SMALL, SEED, "fedavg", verbose=False)
    r2 = run_federated(CLIENTS, SMALL, SEED, "fedavg", verbose=False)
    assert all(np.array_equal(r1.preds[k], r2.preds[k]) for k in r1.preds)


@check("federated: training improves on the untrained model")
def _():
    r = run_federated(CLIENTS, SMALL, SEED, "reliability_fedavg", verbose=False)
    v = r.extra["rounds"]["val_reported"].to_numpy()
    assert v[-1] < v[0]
    return f"reported val MSE {v[0]:.5f} -> {v[-1]:.5f}"


@check("communication accounting: 3 model transfers + gate metadata + heartbeats")
def _():
    r = run_federated(CLIENTS, SMALL, SEED, "reliability_fedavg", event_aware=True, verbose=False)
    rd, mb = r.extra["rounds"], r.extra["model_bytes"]
    hb = len(CLIENTS) - rd["n_active"] - rd["n_dropped"]
    expected = rd["n_active"] * (MODEL_TRANSFERS * mb + METADATA_BYTES) + hb * CFG["event_aware"]["heartbeat_bytes"]
    assert (rd["bytes"] == expected).all() and mb == 7937 * 4
    return f"model update = {mb / 1024:.1f} KB"


@check("faults: each fault degrades only the copy, in the intended way")
def _():
    sdf = generate_virtual_sites(BASE, CFG, 1, SEED)[0][1]
    day = sdf["GHI"] > 50
    st = apply_fault(sdf, "stale", CFG["faults"], SEED, 0)
    assert (st["GHI"].diff() == 0)[day].mean() > 0.8
    bi = apply_fault(sdf, "bias", CFG["faults"], SEED, 0)
    m = sdf["pv"].between(0.05, 0.6)
    assert np.allclose(bi["pv"][m], sdf["pv"][m] * CFG["faults"]["bias"]["scale"])
    tn = apply_fault(sdf, "target_noise", CFG["faults"], SEED, 0)
    assert (tn["pv"] - sdf["pv"])[day].std() > 0.05
    fc = apply_fault(sdf, "feature_corruption", CFG["faults"], SEED, 0)
    assert 0.5 < fc["GHI"][day].mean() / sdf["GHI"][day].mean() < 0.7
    assert np.array_equal(sdf["pv"], generate_virtual_sites(BASE, CFG, 1, SEED)[0][1]["pv"])


@check("faults: baselines train on the degraded copy; healthy metrics exclude faulty sites")
def _():
    cl, _ = prepare_clients(CFG, 4, SEED, faulty_clients=[3], fault_type="stale")
    assert _site_data(cl[3]) is cl[3].faulty and _site_data(cl[0]) is cl[0]
    r = run_method("smart_persistence", cl, CFG, SEED, faulty=[3], verbose=False)
    assert r.extra["faulty_sites"] == ["D"]


def _synthetic_node(n_days=30, sd=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2019-06-01 06:00", periods=n_days * 72, freq="10min")
    f = np.full(len(t), 20.0)
    return pd.DataFrame({"time": t, "split": "test", "daytime": True, "block": 0,
                         "forecast_mw": f, "actual_mw": f + rng.normal(0, sd, len(t)),
                         "demand_mw": 40.0})


@check("reserve: n-sigma is causal (future errors never change today's reserve)")
def _():
    node = _synthetic_node()
    r1, _, _ = nsigma_reserve(node, 0.05, 36, 12, 3)
    node2 = node.copy()
    node2.loc[1000:, "actual_mw"] += 50.0
    r2, _, _ = nsigma_reserve(node2, 0.05, 36, 12, 3)
    assert np.allclose(r1[:1003], r2[:1003]) and not np.allclose(r1[1003:], r2[1003:])


@check("reserve: on Gaussian errors the rule hits its target (implementation is calibrated)")
def _():
    node = _synthetic_node(n_days=200, seed=1)
    out = []
    for dlt in (0.02, 0.05, 0.10):
        r, mu, sd = nsigma_reserve(node, dlt, 36, 12, 3)
        i = 500
        assert abs(r[i] - max(norm.ppf(1 - dlt) * sd[i] - mu[i], 0)) < 1e-9
        known = np.isfinite(mu) & np.isfinite(sd)
        short = (node["actual_mw"].to_numpy() - node["forecast_mw"].to_numpy()) < -r
        cov = 100 * (1 - short[known].mean())
        assert abs(cov - 100 * (1 - dlt)) < 1.5, (dlt, cov)
        out.append(f"{100 * (1 - dlt):.0f}%->{cov:.1f}%")
    return "target->achieved " + ", ".join(out)


@check("reserve: empirical margin is causal and uses an observed tail value")
def _():
    node = _synthetic_node(n_days=10, seed=3)
    r1, q1 = empirical_reserve(node, 0.10, 72, 12, 3)
    node2 = node.copy()
    node2.loc[300:, "actual_mw"] -= 20
    r2, _ = empirical_reserve(node2, 0.10, 72, 12, 3)
    assert np.allclose(r1[:303], r2[:303], equal_nan=True)
    i = np.where(np.isfinite(q1))[0][0]
    past_loss = (node["forecast_mw"] - node["actual_mw"]).shift(3).iloc[:i + 1].dropna().tail(72)
    assert any(abs(q1[i] - x) < 1e-12 for x in past_loss)


@check("calibration: future residuals cannot change current guarded margin or warning")
def _():
    node = _synthetic_node(n_days=4)
    margin, fixed = np.ones(len(node)), np.full(len(node), 5.0)
    a = monitor_margin(node, margin, 0.05, 72, 12, 3, fixed)
    changed = node.copy()
    changed.loc[100:, "actual_mw"] -= 100
    b = monitor_margin(changed, margin, 0.05, 72, 12, 3, fixed)
    assert np.allclose(a.guarded_margin_mw[:103], b.guarded_margin_mw[:103])
    assert np.allclose(a.rolling_coverage[:103], b.rolling_coverage[:103], equal_nan=True)
    assert a.calibration_samples.iloc[0] == 0
    assert a.guarded_margin_mw.iloc[0] == 5.0


@check("scheduler: expected gap, power cap and daily energy cap are enforced")
def _():
    t = pd.date_range("2019-06-01 06:00", periods=12, freq="10min")
    node = pd.DataFrame({"time": t, "split": "test", "daytime": True, "block": 0,
                         "capacity_mw": 10.0, "forecast_mw": 2.0, "actual_mw": 1.0,
                         "demand_mw": 10.0})
    rcfg = copy.deepcopy(CFG["reserve"])
    rcfg.update(grid_import_limit_mw=3.0, backup_power_limit_mw=4.0,
                backup_energy_limit_mwh=2.0)
    d = schedule_backup(node, np.ones(len(node)), rcfg)
    assert np.allclose(d["expected_gap_mw"], 5.0)
    assert (d["scheduled_backup_mw"] <= 4.0 + 1e-9).all()
    assert d["scheduled_backup_mw"].sum() * (10 / 60) <= 2.0 + 1e-8
    assert (d["planned_gap_mw"] > 0).any()


@check("real data path: measured power column + site-ID column -> one client per site")
def _():
    rows = []
    for sid, cap_kw in (("plant_north", 5000.0), ("plant_south", 12000.0)):
        d = BASE.copy()
        d["site"] = sid
        d["power_kw"] = cap_kw * pv_power_pu(d["GHI"], d["Temperature"], 0.85, -0.004,
                                             CFG["pv_model"])
        d["capacity_kw"] = cap_kw
        rows.append(d.drop(columns=["timestamp"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "two_sites.csv"
        pd.concat(rows).to_csv(path, index=False)
        cfg = copy.deepcopy(SMALL)
        cfg["data"].update(path=str(path), site_col="site", target_col=None)
        cfg["target"].update(mode="column", column="power_kw", capacity=None,
                             capacity_col="capacity_kw", column_unit_mw=0.001)
        cl, _ = prepare_clients(cfg, 99, SEED)
        assert [c.params.name for c in cl] == ["plant_north", "plant_south"]
        assert [c.params.capacity_mw for c in cl] == [5.0, 12.0]
        r = run_federated(cl, cfg, SEED, "reliability_fedavg", verbose=False)
        assert all(np.isfinite(p).all() for p in r.preds.values())
    m = measured_pu(pd.DataFrame({"p": [0.0, 50.0, 100.0]}), {"target": {"column": "p"}})
    assert np.allclose(m, [0, 0.5, 1])


fails = [r for r in RESULTS if not r[0]]
print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} checks passed")
import json
audit_path = Path(CFG["paths"]["outputs"]) / "metrics/audit_checks.json"
audit_path.parent.mkdir(parents=True, exist_ok=True)
audit_path.write_text(json.dumps({"passed": len(RESULTS) - len(fails), "total": len(RESULTS),
                                 "checks": [{"passed": ok, "name": name, "detail": detail}
                                            for ok, name, detail in RESULTS]}, indent=2), encoding="utf-8")
sys.exit(1 if fails else 0)
