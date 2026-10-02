"""Reproduce the five-seed constrained reserve benchmark used in the pitch.

Run the five forecast jobs first (seed 42 may use the default c4 tag):
    python run_simulation.py --clients 4 --methods reliability_fedavg --seed 42 --quiet
    python run_simulation.py --clients 4 --methods reliability_fedavg --seed 43 --quiet --tag reserve_seed43
    ...
"""
from pathlib import Path
import argparse

import pandas as pd

from src.data.adapter import load_config
from src.reserve.simulator import load_node, run_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--common", action="store_true", help="aggregate the new common-protocol reserve results")
    args = ap.parse_args()
    cfg = load_config("config.yaml")
    out = Path(cfg["paths"]["outputs"])
    if args.common:
        detail = pd.read_csv(out / "tables/common_reserve_detail.csv")
        detail = detail[detail.forecast_method == "reliability_fedavg"].copy()
        if detail.seed.nunique() != 5:
            raise SystemExit("Common reserve comparison requires five completed seeds")
        metrics = ["reserve_energy_mwh", "shortfall_energy_mwh", "availability_pct",
                   "planned_capacity_gap_mwh", "total_cost"]
        summary = detail.groupby("policy")[metrics].agg(["mean", "std"])
        detail.to_csv(out / "tables/reserve_benchmark_common_detail.csv", index=False)
        summary.to_csv(out / "tables/reserve_benchmark_common_5seed.csv")
        print(summary.round(2).to_string())
        return
    files = {42: out / "tables" / "pred_c4_blocked_monthly_reliability_fedavg.csv"}
    files.update({s: out / "tables" / f"pred_reserve_seed{s}_reliability_fedavg.csv"
                  for s in range(43, 47)})
    rows = []
    for seed, path in files.items():
        if not path.exists():
            raise SystemExit(f"Missing {path}; generate the forecast for seed {seed} first.")
        node = load_node(path)
        for _, summary in run_policies(node, cfg, seed):
            rows.append({"seed": seed, **summary})
    detail = pd.DataFrame(rows)
    metrics = ["reserve_energy_mwh", "shortfall_energy_mwh", "availability_pct",
               "planned_capacity_gap_mwh", "total_cost"]
    aggregate = (detail.groupby(["policy", "delta"], dropna=False)[metrics]
                 .agg(["mean", "std"]).reset_index())
    aggregate.columns = ["_".join(str(v) for v in col if str(v) != "").rstrip("_")
                         if isinstance(col, tuple) else col for col in aggregate.columns]
    detail.to_csv(out / "tables" / "reserve_benchmark_detail.csv", index=False)
    aggregate.to_csv(out / "tables" / "reserve_benchmark_5seed.csv", index=False)
    print("FIVE-SEED CONSTRAINED RESERVE BENCHMARK")
    print(aggregate.round(2).to_string(index=False))
    fixed = aggregate[aggregate["policy"] == "fixed_20pct"].iloc[0]
    chosen = aggregate[aggregate["policy"] == "nsigma_d0.05"].iloc[0]
    print("\nHeadline vs fixed 20% demand margin (means):")
    for metric in ("reserve_energy_mwh", "shortfall_energy_mwh", "total_cost"):
        change = 100 * (chosen[f"{metric}_mean"] / fixed[f"{metric}_mean"] - 1)
        print(f"  {metric}: {change:+.1f}%")
    print("Saved reserve_benchmark_detail.csv and reserve_benchmark_5seed.csv")


if __name__ == "__main__":
    main()
