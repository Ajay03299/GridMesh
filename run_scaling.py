"""Bounded-participation scale simulation from 4 to 500 sites.

    python run_scaling.py                       # N = 4..500 sites, 3 rounds, cap 20
    python run_scaling.py --sizes 4,100 --rounds 10

All methods use the same cap. Protocol stress data above 20 sites repeats four temporal
datasets; its accuracy is illustrative, and geographic validation remains future work.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate
from src.models.forecasting_model import set_seed
from src.evaluation.resources import peak_memory_mb


def prepare_protocol_clients(cfg, n, seed, faulty_indices=()):
    """Share four compact reference datasets across logical clients for protocol stress."""
    from src.data.preprocessing import ClientData, Split
    from src.data.virtual_sites import sample_site_params
    templates, _ = prepare_clients(cfg, 4, seed, [0, 1, 2, 3] if faulty_indices else [], "stale")
    def compact(split, limit):
        take = np.linspace(0, len(split.y) - 1, min(limit, len(split.y)), dtype=int)
        return Split(**{f: getattr(split, f)[take] for f in Split.__dataclass_fields__})
    for t in templates:
        for item in (t, t.faulty):
            if item is not None:
                item.train, item.val, item.test = (compact(item.train, 2000),
                                                  compact(item.val, 1000), compact(item.test, 1000))
    clients = []
    for i in range(n):
        t = templates[i % len(templates)]
        p = sample_site_params(i, cfg["virtual_sites"], seed)
        bad = ClientData(p, t.faulty.train, t.faulty.val, t.faulty.test, t.feature_names,
                         t.faulty.data_quality) if i in faulty_indices else None
        clients.append(ClientData(p, t.train, t.val, t.test, t.feature_names, t.data_quality, bad))
    return clients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,20,50,100,250,500")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-clients-per-round", type=int, default=20)
    ap.add_argument("--max-sites", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    cfg["federated"]["rounds"] = 2 if args.quick else args.rounds
    cfg["federated"]["client_fraction"] = 1.0
    cfg["federated"]["max_clients_per_round"] = args.max_clients_per_round
    seed, out = cfg["seed"], Path(cfg["paths"]["outputs"])
    rows = []
    for n in [int(x) for x in args.sizes.split(",")]:
        if n > args.max_sites:
            continue
        set_seed(seed)
        # Building 500 full year-long feature matrices would measure dataframe duplication, not
        # FL scalability. Above 20 sites we repeat four leakage-safe reference-site datasets with
        # unique identities. This is explicitly a protocol/communication stress test; accuracy
        # at those sizes is illustrative and is never presented as geographic validation.
        if n <= 20:
            clients, _ = prepare_clients(cfg, n, seed)
            data_mode = "full_virtual_sites"
        else:
            clients = prepare_protocol_clients(cfg, n, seed)
            data_mode = "shared_reference_site_protocol_stress"
        methods = ["fedavg", "reliability_fedavg_event"]
        if n >= 100:
            methods.append("hierarchical_reliability_fedavg")
        for m in methods:
            t = time.time()
            res = run_method(m, clients, cfg, seed, [], verbose=False)
            s = evaluate(res, clients)
            rows.append({"n_sites": n, "method": m, "rounds": cfg["federated"]["rounds"],
                         "global_rmse": s["global_rmse"], "worst_site_rmse": s["worst_site_rmse"],
                         "comm_mb_per_round": s["mean_comm_kb_per_round"] / 1000,
                         "participation_rate": s["participation_rate"],
                         "clients_per_round_cap": args.max_clients_per_round,
                         "data_mode": data_mode,
                         "sites_per_round": float(res.extra["rounds"].n_active.mean()),
                         "total_comm_mb": s["total_comm_mb"],
                         "feeder_comm_mb_per_round": float(res.extra["rounds"].feeder_bytes.mean() / 1e6),
                         "rounds_to_converge": s["rounds_to_converge"],
                         "client_training_seconds": res.extra["training_seconds"],
                         "coordinator_seconds": res.extra["coordinator_seconds"],
                         "inference_seconds": res.extra["inference_seconds"],
                         "process_peak_memory_mb": peak_memory_mb(),
                         "seconds": round(time.time() - t, 1)})
            r = rows[-1]
            print(f"  N={n:>3} {m:<26} RMSE {r['global_rmse']:.4f} | "
                  f"{r['comm_mb_per_round']:.2f} MB/round | participation "
                  f"{100 * r['participation_rate']:.0f}% | {r['seconds']}s")
        del clients
    (out / "tables").mkdir(parents=True, exist_ok=True)
    filename = "comm_scaling_quick.csv" if args.quick else "comm_scaling.csv"
    pd.DataFrame(rows).to_csv(out / "tables" / filename, index=False)
    import json
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "metrics" / filename.replace(".csv", ".json")).write_text(
        json.dumps({"args": vars(args), "config": cfg, "results": rows}, indent=2))
    from src.visualization.plots import plot_scaling
    if not args.quick:
        plot_scaling(out)
    # The legacy report combines old experiment tables. New evidence gets its own report
    # after all paired runs complete, rather than silently mixing training protocols.
    print("Saved outputs/tables/comm_scaling.csv and outputs/plots/14_comm_scaling.png")


if __name__ == "__main__":
    main()
