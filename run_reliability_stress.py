"""Reproducible resilience scorecard across faults, dropout and under-participation."""
import argparse
import copy
from pathlib import Path
import time

import pandas as pd

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate
from src.federated.server import MODEL_TRANSFERS


SCENARIOS = [
    {"name": "healthy", "faulty": [], "fault": None, "severity": 1, "dropout": 0.0, "fraction": 1.0},
    {"name": "stale_50pct", "faulty": [2, 3], "fault": "stale", "severity": 2, "dropout": 0.0, "fraction": 1.0},
    {"name": "biased_25pct", "faulty": [3], "fault": "bias", "severity": 2, "dropout": 0.0, "fraction": 1.0},
    {"name": "dropout_50pct", "faulty": [], "fault": None, "severity": 1, "dropout": 0.5, "fraction": 1.0},
    {"name": "under_participation", "faulty": [], "fault": None, "severity": 1, "dropout": 0.25, "fraction": 0.5},
    {"name": "fault_and_dropout", "faulty": [3], "fault": "stale", "severity": 2, "dropout": 0.25, "fraction": 1.0},
    {"name": "sensor_recovery", "faulty": [3], "fault": "stale", "severity": 2, "dropout": 0.0, "fraction": 1.0},
    {"name": "seasonal_shift", "faulty": [], "fault": None, "severity": 1, "dropout": 0.0, "fraction": 1.0},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--scenarios", default="all", help="comma-separated scenario names or all")
    args = ap.parse_args()
    base, rows = load_config(), []
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else \
        [42] if args.quick else base["experiments"]["seeds"]
    for seed in seeds:
        for sc in SCENARIOS:
            if args.scenarios != "all" and sc["name"] not in args.scenarios.split(","):
                continue
            cfg = copy.deepcopy(base)
            cfg["federated"]["rounds"] = 5 if args.quick else 20
            cfg["federated"]["dropout_rate"] = sc["dropout"]
            cfg["federated"]["client_fraction"] = sc["fraction"]
            if sc["name"] == "seasonal_shift":
                cfg["split"]["mode"] = "chronological"
            if sc["name"] == "sensor_recovery":
                cfg["faults"]["recovery_round"] = 4 if args.quick else 10
            clients, _ = prepare_clients(cfg, 4, seed, sc["faulty"], sc["fault"], sc["severity"])
            for method in ("fedavg", "reliability_fedavg", "reliability_fedavg_event"):
                started = time.perf_counter()
                result = run_method(method, clients, cfg, seed, sc["faulty"], verbose=False)
                metric = evaluate(result, clients)
                rd = result.extra["rounds"]
                log = pd.DataFrame(result.history)
                recovery_rounds = float("nan")
                if sc["name"] == "sensor_recovery":
                    recovered = log[(log.site == clients[3].params.name) &
                                    (log["round"] >= cfg["faults"]["recovery_round"]) &
                                    (log.weight > 0) & (~log.quarantined)]
                    if not recovered.empty:
                        recovery_rounds = int(recovered["round"].min() - cfg["faults"]["recovery_round"])
                rows.append({"seed": seed, "scenario": sc["name"], "method": method,
                             "rounds": cfg["federated"]["rounds"],
                             "communication_model_transfers": MODEL_TRANSFERS,
                             "global_rmse": metric["global_rmse"],
                             "healthy_rmse": metric["healthy_rmse"],
                             "worst_site_rmse": metric["worst_site_rmse"],
                             "forecast_bias": metric["global_bias"],
                             "participation_rate": metric["participation_rate"],
                             "comm_mb": metric["total_comm_mb"],
                             "rejected_updates": int(rd["n_rejected"].sum()),
                             "fallback_rounds": int((rd["fallback"] != "").sum()),
                             "quarantined_sites": int(log.loc[log.quarantined, "site"].nunique()),
                             "drift_events": int(log.drift.sum()),
                             "dropped_events": int(rd.n_dropped.sum()),
                             "quarantine_weight_recovery_rounds": recovery_rounds,
                             "communication_availability": float(1 - rd.n_dropped.sum() /
                                     max(rd.n_active.sum() + rd.n_dropped.sum(), 1)),
                             "runtime_seconds": time.perf_counter() - started})
            print(f"finished {sc['name']} seed={seed}")
    out = Path(base["paths"]["outputs"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    prefix = "reliability_quick" if args.quick else "reliability"
    if args.scenarios != "all":
        prefix += "_" + args.scenarios.replace(",", "_")
    detail.to_csv(out / f"tables/{prefix}_stress_detail.csv", index=False)
    summary = detail.groupby(["scenario", "method"], as_index=False).agg(
        healthy_rmse=("healthy_rmse", "mean"), worst_site_rmse=("worst_site_rmse", "mean"),
        participation=("participation_rate", "mean"), comm_mb=("comm_mb", "mean"),
        fallback_rounds=("fallback_rounds", "mean"))
    summary.to_csv(out / f"tables/{prefix}_scorecard.csv", index=False)
    import json
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / f"metrics/{prefix}.json").write_text(json.dumps({"quick": args.quick,
        "seeds": seeds, "config": base, "scenarios": SCENARIOS, "results": rows}, indent=2))
    print("\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
