"""Paired protocol stress tests, including an unavailable neighbourhood."""
import argparse
import copy
from src.evaluation.serialization import dumps
from pathlib import Path

import pandas as pd

from run_scaling import prepare_protocol_clients
from run_simulation import run_method
from src.data.adapter import load_config
from src.evaluation.experiments import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    base, rows = load_config(), []
    scenarios = [("healthy", 0.0, 0.0), ("dropout25", 0.25, 0.0),
                 ("dropout50", 0.50, 0.0), ("fault10", 0.0, 0.10),
                 ("fault20", 0.0, 0.20), ("fault20_dropout25", 0.25, 0.20),
                 ("neighbourhood_offline", 0.0, 0.0)]
    for scenario, dropout, fault_share in scenarios:
        cfg = copy.deepcopy(base)
        cfg["federated"].update(rounds=args.rounds, dropout_rate=dropout, max_clients_per_round=20)
        faults = list(range(int(args.sites * fault_share)))
        clients = prepare_protocol_clients(cfg, args.sites, args.seed, faults)
        if scenario == "neighbourhood_offline":
            # The first neighbourhood disappears from communication. Keep all sites in
            # forecast evaluation so loss of access cannot improve the reported denominator.
            cfg["federated"]["unavailable_sites"] = [c.params.name for c in clients[:25]]
        for method in ["fedavg", "reliability_fedavg", "reliability_fedavg_event",
                       "hierarchical_reliability_fedavg"]:
            result = run_method(method, clients, cfg, args.seed, faults, verbose=False)
            metric = evaluate(result, clients)
            rows.append({"scenario": scenario, "sites": args.sites, "seed": args.seed,
                         "rounds": args.rounds, "method": method,
                         "healthy_rmse": metric["healthy_rmse"],
                         "global_rmse": metric["global_rmse"],
                         "worst_site_rmse": metric["worst_site_rmse"],
                         "communication_mb": metric["total_comm_mb"],
                         "feeder_communication_mb": float(result.extra["rounds"].feeder_bytes.sum()/1e6),
                         "fallback_rounds": int((result.extra["rounds"].fallback != "").sum()),
                         "data_mode": "shared_reference_site_protocol_stress"})
        print(f"finished {scenario}", flush=True)
    out = Path(base["paths"]["outputs"])
    pd.DataFrame(rows).to_csv(out / "tables/scale_stress.csv", index=False)
    (out / "metrics/scale_stress.json").write_text(dumps(
        {"args": vars(args), "config": base, "results": rows}, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
