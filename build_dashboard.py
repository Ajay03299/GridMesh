"""Pre-compute the dashboard scenarios (run once, ~30 s on a MacBook), then open the dashboard.

    python build_dashboard.py
    streamlit run dashboard.py

Scenarios: normal | faulty (stale sensors at sites C and D) | dropout (25%) |
shift (trained Jan-Sep, tested Nov-Dec = weather/regime shift).
Each scenario trains FedAvg, reliability-aware FedAvg and GridMesh (reliability-aware +
event-aware FL); the GridMesh forecasts feed the reserve node. Output: outputs/dashboard/<scenario>/
"""
import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate, predictions_frame
from src.models.forecasting_model import set_seed
from src.reserve.simulator import load_node, run_policies

OURS = "reliability_fedavg_event"
SCENARIOS = {
    "normal": {"title": "Normal operation", "faulty": [], "fault_type": None,
               "dropout": 0.0, "split": None},
    "faulty": {"title": "Faulty sensors at sites C and D (frozen / stale readings)",
               "faulty": [2, 3], "fault_type": "stale", "dropout": 0.0, "split": None},
    "dropout": {"title": "25% of selected sites randomly drop out each round", "faulty": [],
                "fault_type": None, "dropout": 0.25, "split": None},
    "shift": {"title": "Weather / regime shift: trained on Jan-Sep, tested on Nov-Dec",
              "faulty": [], "fault_type": None, "dropout": 0.0, "split": "chronological"},
}


def node_series(res, clients, cfg, seed):
    """Reserve node time series (test period) for one forecast result."""
    runs = {s["policy"]: (ts, s) for ts, s in
            run_policies(load_node(predictions_frame(res, clients)), cfg, seed)}
    return runs


def build(name, sc, base_cfg, out, n_clients):
    cfg = copy.deepcopy(base_cfg)
    cfg["federated"]["dropout_rate"] = sc["dropout"]
    if sc["split"]:
        cfg["split"]["mode"] = sc["split"]
    seed = cfg["seed"]
    set_seed(seed)
    clients, split_info = prepare_clients(cfg, n_clients, seed, sc["faulty"], sc["fault_type"])
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    kpi = {"scenario": name, "title": sc["title"], "test_period": split_info["test"],
           "faulty_sites": [clients[i].params.name for i in sc["faulty"]],
           "dropout": sc["dropout"], "delta": cfg["reserve"]["delta"]}

    for m in ["smart_persistence", "fedavg", "reliability_fedavg", OURS]:
        res = run_method(m, clients, cfg, seed, sc["faulty"], verbose=False)
        s = evaluate(res, clients)
        kpi[m] = {k: s.get(k) for k in ("global_rmse", "healthy_rmse", "worst_site_rmse",
                                        "total_comm_mb", "participation_rate")}
        if m != "smart_persistence":
            pd.DataFrame(res.history).to_csv(d / f"clients_{m}.csv", index=False)
            res.extra["rounds"].to_csv(d / f"rounds_{m}.csv", index=False)
        if m == OURS:
            kpi["forecast_health"] = res.extra["forecast_health"]
            rd = res.extra["rounds"]
            kpi["safety"] = {"rejected_updates": int(rd.n_rejected.sum()),
                             "fallback_rounds": int((rd.fallback != "").sum()),
                             "rollback_rounds": int((rd.rollback_reason != "").sum())}
        runs = node_series(res, clients, cfg, seed)
        fixed_key = f"fixed_{int(round(cfg['reserve']['fixed_fraction'] * 100))}pct"
        ns_key = f"guarded_nsigma_d{cfg['reserve']['delta']}"
        if m == "smart_persistence":
            sp_forecast = runs[fixed_key][0][["time", "forecast_mw"]].rename(
                columns={"forecast_mw": "forecast_sp_mw"})
        if m == OURS:
            ns, fx = runs[ns_key][0], runs[fixed_key][0]
            node = ns[["time", "split", "daytime", "capacity_mw", "forecast_mw", "actual_mw",
                       "demand_mw", "grid_import_mw", "expected_gap_mw",
                       "uncertainty_margin_mw", "required_backup_mw", "planned_gap_mw",
                       "deficit_mw", "reserve_mw", "shortfall_mw", "mu_e", "sigma_e",
                       "rolling_coverage", "calibration_samples", "calibration_age_minutes",
                       "margin_source", "calibration_warning", "forecast_age_minutes",
                       "forecast_warning"]]
            node = node.assign(backup_power_limit_mw=ns['backup_power_limit_mw'].to_numpy(),
                               backup_energy_limit_mwh=ns['backup_energy_limit_mwh'].to_numpy(),
                               step_hours=ns['step_hours'].to_numpy())
            node = node.assign(reserve_fixed_mw=fx["reserve_mw"].to_numpy(),
                               shortfall_fixed_mw=fx["shortfall_mw"].to_numpy())
            kpi["reserve"] = {k: {x: v[1][x] for x in ("reserve_pct_of_forecast",
                                                       "ens_pct_of_demand", "availability_pct")}
                              for k, v in runs.items()}
    node = node.merge(sp_forecast, on="time", how="left")
    node[node.split == "test"].to_csv(d / "node.csv", index=False)
    pd.DataFrame([{"site": c.params.name, "capacity_mw": c.params.capacity_mw,
                   "faulty": i in sc["faulty"]} for i, c in enumerate(clients)]).to_csv(
        d / "sites.csv", index=False)
    (d / "kpi.json").write_text(json.dumps(kpi, indent=2, default=str))
    return kpi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    out = Path(cfg["paths"]["outputs"]) / "dashboard"
    for name, sc in SCENARIOS.items():
        t = time.time()
        k = build(name, sc, cfg, out, args.clients)
        print(f"  {name:<8} healthy-site RMSE  FedAvg {k['fedavg']['healthy_rmse']:.4f} | "
              f"GridMesh {k[OURS]['healthy_rmse']:.4f} | smart persistence "
              f"{k['smart_persistence']['healthy_rmse']:.4f}   ({time.time() - t:.0f}s)")
    print(f"Saved to {out}.  Next:  streamlit run dashboard.py")


if __name__ == "__main__":
    main()
