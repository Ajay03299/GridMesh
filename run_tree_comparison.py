"""Fair tree-model comparison on the exact GridMesh split, features and residual target."""
import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd

from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import (evaluate, run_centralized_xgboost, run_decision_tree,
                                        run_local_xgboost, run_persistence)
from src.models.tree_models import feature_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds; full defaults to five")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg, rows, importance = load_config(), [], []
    out = Path(cfg["paths"]["outputs"])
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else \
        [42] if args.quick else cfg["experiments"]["seeds"]
    prefix = "tree_quick" if args.quick else "tree_comparison"
    for seed in seeds:
        clients, split = prepare_clients(cfg, args.clients, seed)
        methods = [run_persistence(clients, False), run_persistence(clients, True)]
        runners = [run_decision_tree, run_local_xgboost, run_centralized_xgboost]
        for runner in runners:
            started = time.perf_counter()
            result = runner(clients, cfg, seed, args.quick) if "xgboost" in runner.__name__ \
                else runner(clients, cfg, seed)
            result.extra.setdefault("training_seconds", time.perf_counter() - started)
            methods.append(result)
        for result in methods:
            metric = evaluate(result, clients)
            row = {k: v for k, v in metric.items() if k != "per_site"}
            row.update(seed=seed, clients=args.clients,
                       training_seconds=result.extra.get("training_seconds", 0.0),
                       inference_seconds=result.extra.get("inference_seconds", 0.0),
                       model_mb=result.extra.get("model_bytes", 0) / 1e6,
                       data_arrangement=result.extra.get("data_arrangement", "no training"))
            rows.append(row)
            for feature, value in result.extra.get("feature_importance", {}).items():
                importance.append({"seed": seed, "method": result.method,
                                   "feature": feature, "importance": value})
        print(f"seed {seed}: " + ", ".join(f"{r.method}={evaluate(r, clients)['global_rmse']:.4f}"
                                             for r in methods))
    detail = pd.DataFrame(rows)
    summary = detail.groupby("method", as_index=False).agg(
        rmse_mean=("global_rmse", "mean"), rmse_std=("global_rmse", "std"),
        mae_mean=("global_mae", "mean"), worst_site_rmse=("worst_site_rmse", "mean"),
        training_seconds=("training_seconds", "mean"),
        inference_seconds=("inference_seconds", "mean"), model_mb=("model_mb", "mean"))
    detail.to_csv(out / f"tables/{prefix}_detail.csv", index=False)
    summary.to_csv(out / f"tables/{prefix}_summary.csv", index=False)
    imp = pd.DataFrame(importance)
    imp.to_csv(out / f"tables/{prefix}_feature_importance.csv", index=False)
    if not imp.empty:
        grouped = imp.assign(group=imp.feature.map(feature_group)).groupby(
            ["seed", "method", "group"], as_index=False).importance.sum()
        grouped.to_csv(out / f"tables/{prefix}_importance_groups.csv", index=False)
    if not imp.empty:
        top = (imp.groupby(["method", "feature"], as_index=False).importance.mean()
               .sort_values(["method", "importance"], ascending=[True, False])
               .groupby("method").head(10))
        fig, axes = plt.subplots(1, top.method.nunique(), figsize=(12, 5), squeeze=False)
        for ax, (method, frame) in zip(axes[0], top.groupby("method")):
            frame = frame.sort_values("importance")
            ax.barh(frame.feature, frame.importance, color="#12A86B")
            ax.set_title(method.replace("_", " ").title())
        fig.tight_layout()
        fig.savefig(out / "plots/tree_feature_importance.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    with open(out / f"metrics/{prefix}.json", "w") as f:
        json.dump({"quick": args.quick, "clients": args.clients, "split": split,
                   "results": rows}, f, indent=2, default=str)
    print("\n" + summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
