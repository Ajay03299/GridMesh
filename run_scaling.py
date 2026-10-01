"""Scale simulation: communication and accuracy from 4 to 100 sites.

    python run_scaling.py                       # N = 4, 20, 50, 100 sites, 5 rounds
    python run_scaling.py --sizes 4,100 --rounds 10

Every site is eligible every round (client_fraction 1.0), so FedAvg's traffic grows with N;
event-aware participation lets stable sites skip. Writes outputs/tables/comm_scaling.csv
and outputs/plots/14_comm_scaling.png, then refreshes outputs/RESULTS.md.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate
from src.models.forecasting_model import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,20,50,100")
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    cfg["federated"]["rounds"] = args.rounds
    cfg["federated"]["client_fraction"] = 1.0
    seed, out = cfg["seed"], Path(cfg["paths"]["outputs"])
    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        set_seed(seed)
        clients, _ = prepare_clients(cfg, n, seed)
        for m in ["fedavg", "reliability_fedavg_event"]:
            t = time.time()
            res = run_method(m, clients, cfg, seed, [], verbose=False)
            s = evaluate(res, clients)
            rows.append({"n_sites": n, "method": m, "rounds": args.rounds,
                         "global_rmse": s["global_rmse"], "worst_site_rmse": s["worst_site_rmse"],
                         "comm_mb_per_round": s["mean_comm_kb_per_round"] / 1000,
                         "participation_rate": s["participation_rate"],
                         "seconds": round(time.time() - t, 1)})
            r = rows[-1]
            print(f"  N={n:>3} {m:<26} RMSE {r['global_rmse']:.4f} | "
                  f"{r['comm_mb_per_round']:.2f} MB/round | participation "
                  f"{100 * r['participation_rate']:.0f}% | {r['seconds']}s")
        del clients
    (out / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tables" / "comm_scaling.csv", index=False)
    from src.visualization.plots import plot_scaling
    plot_scaling(out)
    from src.evaluation.report import write_results_md
    if (out / "tables" / "comparison_main.csv").exists():
        write_results_md(out, cfg)
    print("Saved outputs/tables/comm_scaling.csv and outputs/plots/14_comm_scaling.png")


if __name__ == "__main__":
    main()
