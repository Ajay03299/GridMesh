"""GridMesh main entry point.

Examples:
    python run_simulation.py --clients 4 --methods baselines
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import (evaluate, run_centralized, run_local_only,
                                        run_persistence, save_predictions)
from src.federated.server import run_federated
from src.models.forecasting_model import set_seed

BASELINES = ["persistence", "smart_persistence", "local_only", "centralized"]
FL = ["fedavg", "reliability_fedavg", "reliability_fedavg_event"]
METHOD_GROUPS = {"baselines": BASELINES, "fl": FL, "all": BASELINES + FL}


def parse_args():
    ap = argparse.ArgumentParser(description="Reliability-aware federated PV forecasting")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--rounds", type=int, default=None, help="override federated.rounds")
    ap.add_argument("--faulty-client", default=None,
                    help="site index (0-based) or letter, comma list allowed: D | 3 | C,D")
    ap.add_argument("--faulty-frac", type=float, default=None,
                    help="make this share of sites faulty (picked at random with the seed)")
    ap.add_argument("--fault-severity", type=float, default=1.0,
                    help="0 = no fault, 1 = config values, 2 = twice as strong")
    ap.add_argument("--fault-type", default="stale",
                    choices=["feature_corruption", "target_noise", "stale", "bias"])
    ap.add_argument("--fault-start", type=int, default=None,
                    help="round at which the fault appears (override faults.start_round)")
    ap.add_argument("--dropout-rate", type=float, default=None)
    ap.add_argument("--client-fraction", type=float, default=None)
    ap.add_argument("--quiet", action="store_true", help="hide per-round logs")
    ap.add_argument("--methods", default="all",
                    help="comma list or group: " + ", ".join(METHOD_GROUPS))
    ap.add_argument("--split", choices=["blocked_monthly", "chronological"], default=None,
                    help="override split.mode (chronological = seasonal-shift stress test)")
    ap.add_argument("--tag", default=None, help="name for this run's output files")
    ap.add_argument("--architecture", choices=["mlp", "gru", "lstm"], default=None,
                    help="override model.architecture")
    return ap.parse_args()


def parse_sites(value, n_clients, frac, seed):
    """'D' / '3' / 'C,D' -> [3] / [3] / [2, 3];  or a random `frac` of all sites."""
    if frac:
        import numpy as np
        k = max(1, int(round(frac * n_clients)))
        return sorted(np.random.default_rng([seed, 3]).choice(n_clients, k, replace=False).tolist())
    if not value:
        return []
    out = []
    for v in value.split(","):
        v = v.strip()
        out.append(int(v) if v.isdigit() else ord(v.upper()) - ord("A"))
    return sorted(out)


def run_method(name, clients, cfg, seed, faulty=None, verbose=True):
    res = _run(name, clients, cfg, seed, faulty, verbose)
    # healthy-site metrics exclude the faulty sites for EVERY method, baselines included
    res.extra.setdefault("faulty_sites", sorted(clients[i].params.name for i in (faulty or [])))
    return res


def _run(name, clients, cfg, seed, faulty=None, verbose=True):
    if name == "persistence":
        return run_persistence(clients, smart=False)
    if name == "smart_persistence":
        return run_persistence(clients, smart=True)
    if name == "local_only":
        return run_local_only(clients, cfg, seed)
    if name == "centralized":
        return run_centralized(clients, cfg, seed)
    if name in FL:
        return run_federated(clients, cfg, seed,
                             method="fedavg" if name == "fedavg" else "reliability_fedavg",
                             event_aware=name.endswith("_event"),
                             faulty_clients=faulty, verbose=verbose)
    raise ValueError(f"Unknown method '{name}'")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.architecture:
        cfg["model"]["architecture"] = args.architecture
    if args.split:
        cfg["split"]["mode"] = args.split
    for key, val in (("rounds", args.rounds), ("dropout_rate", args.dropout_rate),
                     ("client_fraction", args.client_fraction)):
        if val is not None:
            cfg["federated"][key] = val
    if args.fault_start is not None:
        cfg["faults"]["start_round"] = args.fault_start
    seed = args.seed if args.seed is not None else cfg["seed"]
    faulty = parse_sites(args.faulty_client, args.clients, args.faulty_frac, seed)
    set_seed(seed)
    methods = [m for part in args.methods.split(",") for m in METHOD_GROUPS.get(part, [part])]
    arch_suffix = "" if cfg["model"]["architecture"] == "mlp" else f"_{cfg['model']['architecture']}"
    tag = args.tag or (f"c{args.clients}_{cfg['split']['mode']}{arch_suffix}"
                       + (f"_fault{len(faulty)}-{args.fault_type}-s{args.fault_severity:g}"
                          if faulty else "")
                       + (f"_drop{cfg['federated']['dropout_rate']}"
                          if cfg["federated"]["dropout_rate"] else ""))
    out = Path(cfg["paths"]["outputs"])
    for sub in ("metrics", "tables", "logs", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    clients, split_info = prepare_clients(cfg, n_clients=args.clients, seed=seed,
                                          faulty_clients=faulty, fault_type=args.fault_type,
                                          severity=args.fault_severity)
    if faulty:
        print(f"FAULTY sites ({args.fault_type}, severity {args.fault_severity:g}, from round "
              f"{cfg['faults']['start_round']}): {[clients[i].params.name for i in faulty]}")
    print(f"Prepared {len(clients)} virtual sites in {time.time() - t0:.1f}s | "
          f"split={split_info['mode']} | test {split_info['test']}")

    summaries = []
    for name in methods:
        t = time.time()
        if name in FL and not args.quiet:
            print(f"  {name}:")
        res = run_method(name, clients, cfg, seed, faulty, verbose=not args.quiet)
        s = evaluate(res, clients)
        s["seconds"] = round(time.time() - t, 1)
        summaries.append(s)
        save_predictions(res, clients, out / "tables" / f"pred_{tag}_{name}.csv")
        if res.history and name in FL:
            pd.DataFrame(res.history).to_csv(out / "logs" / f"clients_{tag}_{name}.csv", index=False)
            res.extra["rounds"].to_csv(out / "logs" / f"rounds_{tag}_{name}.csv", index=False)
        print(f"  {name:<26} done in {s['seconds']:>5}s")

    table = pd.DataFrame([{k: v for k, v in s.items() if k != "per_site"} for s in summaries])
    print("\nTEST RESULTS — daytime targets, p.u. of installed capacity (lower is better)")
    cols = ["method", "global_mae", "global_rmse", "global_bias", "worst_site_rmse", "worst_site"]
    if faulty:
        cols += ["healthy_rmse"]
    if "total_comm_mb" in table:
        cols += ["rounds_to_converge", "participation_rate", "total_comm_mb"]
    print(table[cols].round(4).to_string(index=False))
    table.to_csv(out / "tables" / f"results_{tag}.csv", index=False)

    with open(out / "metrics" / f"run_{tag}.json", "w") as f:
        json.dump({"args": vars(args), "seed": seed, "config": cfg,
                   "split": split_info,
                   "sites": [vars(c.params) | {"n_train": c.n_train,
                                                "data_quality": c.data_quality}
                             for c in clients],
                   "results": summaries}, f, indent=2, default=str)
    print(f"\nSaved: outputs/tables/results_{tag}.csv, outputs/metrics/run_{tag}.json")


if __name__ == "__main__":
    main()
