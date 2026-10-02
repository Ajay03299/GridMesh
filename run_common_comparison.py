"""Freeze a common protocol, compare all model families, then report reserve outcomes.

Quick: one seed, three FL rounds / neural epochs, 250 boosting rounds.
Full: five paired seeds, config training budgets. Model ranking uses validation only.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import (evaluate, predictions_frame, run_local_xgboost,
                                        run_centralized_xgboost)
from src.reserve.simulator import load_node, run_policies
from src.federated.server import MODEL_TRANSFERS


METHODS = ["persistence", "smart_persistence", "decision_tree", "local_xgboost",
           "centralized_xgboost", "local_only", "centralized", "fedavg",
           "reliability_fedavg", "reliability_fedavg_event", "gru", "lstm"]


def validation_rmse(result, clients):
    errors = np.concatenate([(c.val.y[c.val.daytime] -
                              result.val_preds[c.params.name][c.val.daytime]) for c in clients])
    return float(np.sqrt(np.mean(errors ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--skip-reserve", action="store_true")
    args = ap.parse_args()
    base = load_config()
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else \
        [42] if args.quick else base["experiments"]["seeds"]
    prefix = "common_quick" if args.quick else "common"
    out = Path(base["paths"]["outputs"])
    for folder in ("tables", "metrics"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    rows, per_site, reserve_rows, rank_rows = [], [], [], []
    for seed in seeds:
        cfg = copy.deepcopy(base)
        if args.quick:
            cfg["model"]["epochs"] = cfg["federated"]["rounds"] = 3
        clients, split = prepare_clients(cfg, 4, seed)
        results = {}
        for method in args.methods.split(","):
            mc = copy.deepcopy(cfg)
            started = time.perf_counter()
            if method in ("gru", "lstm"):
                mc["model"]["architecture"] = method
                result = run_method("reliability_fedavg", clients, mc, seed, verbose=False)
                result.method = method
            elif method == "local_xgboost":
                result = run_local_xgboost(clients, mc, seed, args.quick)
            elif method == "centralized_xgboost":
                result = run_centralized_xgboost(clients, mc, seed, args.quick)
            else:
                result = run_method(method, clients, mc, seed, verbose=False)
            elapsed = time.perf_counter() - started
            metric = evaluate(result, clients)
            vrmse = validation_rmse(result, clients)
            row = {k: v for k, v in metric.items() if k != "per_site"}
            row.update(seed=seed, validation_rmse=vrmse, runtime_seconds=elapsed,
                       communication_model_transfers=MODEL_TRANSFERS,
                       training_seconds=result.extra.get("training_seconds", elapsed),
                       inference_seconds=result.extra.get("inference_seconds", np.nan),
                       model_bytes=result.extra.get("model_bytes", np.nan),
                       data_arrangement=result.extra.get("data_arrangement",
                          "federated parameters" if "rounds" in result.extra else
                          "pooled raw data" if method == "centralized" else "local only"))
            rows.append(row)
            per_site.extend({"seed": seed, "method": method, "site": name, **m}
                            for name, m in metric["per_site"].items())
            results[method] = result
            print(f"seed={seed} {method:28} val={vrmse:.4f} test={metric['global_rmse']:.4f} "
                  f"{elapsed:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / f"tables/{prefix}_detail.csv", index=False)
        # Practical comparison candidates are fixed before test evaluation. Ranking records
        # validation evidence; test reserve performance never chooses the forecast model.
        practical = [m for m in ("smart_persistence", "local_xgboost",
                                 "reliability_fedavg", "reliability_fedavg_event") if m in results]
        ranked = sorted(practical, key=lambda m: validation_rmse(results[m], clients))
        rank_rows.extend({"seed": seed, "method": m, "validation_rank": i + 1,
                          "validation_rmse": validation_rmse(results[m], clients)}
                         for i, m in enumerate(ranked))
        if not args.skip_reserve:
            for method in practical:
                node = load_node(predictions_frame(results[method], clients))
                rc = copy.deepcopy(cfg)
                rc["reserve"]["delta_sweep"] = [0.05]
                for _, metric in run_policies(node, rc, seed):
                    reserve_rows.append({"seed": seed, "forecast_method": method, **metric})
                print(f"reserve finished: {method} seed={seed}", flush=True)
            pd.DataFrame(reserve_rows).to_csv(out / f"tables/{prefix}_reserve_detail.csv", index=False)
    detail = pd.DataFrame(rows)
    summary = detail.groupby("method", as_index=False).agg(
        rmse_mean=("global_rmse", "mean"), rmse_std=("global_rmse", "std"),
        mae_mean=("global_mae", "mean"), bias_mean=("global_bias", "mean"),
        worst_site_rmse=("worst_site_rmse", "mean"), validation_rmse=("validation_rmse", "mean"),
        runtime_seconds=("runtime_seconds", "mean"), model_bytes=("model_bytes", "mean"))
    summary.to_csv(out / f"tables/{prefix}_summary.csv", index=False)
    pd.DataFrame(per_site).to_csv(out / f"tables/{prefix}_per_site.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(out / f"tables/{prefix}_validation_selection.csv", index=False)
    if reserve_rows:
        r = pd.DataFrame(reserve_rows)
        metrics = ["reserve_energy_mwh", "shortfall_energy_mwh", "availability_pct",
                   "planned_capacity_gap_mwh", "grid_import_energy_mwh", "total_cost"]
        r.groupby(["forecast_method", "policy"])[metrics].agg(["mean", "std"]).to_csv(
            out / f"tables/{prefix}_reserve_summary.csv")
    (out / f"metrics/{prefix}.json").write_text(json.dumps(
        {"quick": args.quick, "seeds": seeds, "config": cfg, "split": split,
         "methods": args.methods.split(","), "results": rows}, indent=2, default=str))
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
