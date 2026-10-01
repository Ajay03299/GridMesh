"""Phase 9 — one-node reserve simulation: fixed 20% vs n-sigma forecast-aware reserve.

Run run_simulation.py first (it writes the forecast files this script reads).

Examples:
    python run_reserve_sim.py                      # both policies + delta sweep
    python run_reserve_sim.py --policy fixed20
    python run_reserve_sim.py --policy nsigma --delta 0.02
    python run_reserve_sim.py --method fedavg      # use another method's forecasts
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.adapter import load_config
from src.reserve.reserve_policy import fixed_reserve, nsigma_reserve
from src.reserve.simulator import load_node, simulate, synthetic_demand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--policy", choices=["fixed20", "nsigma", "both"], default="both")
    ap.add_argument("--delta", type=float, default=None, help="n-sigma shortfall probability")
    ap.add_argument("--method", default=None, help="forecast method (default: config)")
    ap.add_argument("--run-tag", default="c4_blocked_monthly",
                    help="tag of the run_simulation.py outputs to read")
    ap.add_argument("--pred", default=None, help="explicit path to a pred_*.csv file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rcfg, out = cfg["reserve"], Path(cfg["paths"]["outputs"])
    method = args.method or rcfg["forecast_method"]
    pred = Path(args.pred or out / "tables" / f"pred_{args.run_tag}_{method}.csv")
    if not pred.exists():
        raise SystemExit(f"Missing {pred}. Run: python run_simulation.py --clients 4 --quiet")

    node = load_node(pred)
    node["demand_mw"] = synthetic_demand(node["time"], node["capacity_mw"].iloc[0],
                                         rcfg["demand"], cfg["seed"])
    h = cfg["features"]["horizon"]
    print(f"Node: {node['n_sites'].iloc[0]} sites, {node['capacity_mw'].iloc[0]:.1f} MW installed "
          f"(simulated capacities) | forecasts from '{method}' | demand is SYNTHETIC")

    runs = []
    if args.policy in ("fixed20", "both"):
        r = fixed_reserve(node["forecast_mw"], rcfg["fixed_fraction"])
        runs.append(simulate(node, r, rcfg, f"fixed_{int(rcfg['fixed_fraction'] * 100)}pct"))
    if args.policy in ("nsigma", "both"):
        deltas = [args.delta] if args.delta else (
            rcfg["delta_sweep"] if args.policy == "both" else [rcfg["delta"]])
        for dlt in deltas:
            r, mu, sd = nsigma_reserve(node, dlt, rcfg["window"], rcfg["min_periods"], h)
            d, s = simulate(node, r, rcfg, f"nsigma_d{dlt}", delta=dlt)
            d["mu_e"], d["sigma_e"] = mu, sd
            runs.append((d, s))

    for d, s in runs:
        d.to_csv(out / "tables" / f"reserve_ts_{s['policy']}.csv", index=False)
    table = pd.DataFrame([s for _, s in runs])
    show = ["policy", "reserve_energy_mwh", "shortfall_energy_mwh", "ens_pct_of_demand",
            "reserve_pct_of_forecast", "availability_pct", "target_availability_pct", "total_cost"]
    print(f"\nRESERVE RESULTS — daytime test intervals (n={table['intervals'].iloc[0]}), "
          f"cost coefficients are SIMULATION ASSUMPTIONS")
    print(table[show].round(2).to_string(index=False))
    table.to_csv(out / "tables" / "reserve_summary.csv", index=False)
    with open(out / "metrics" / "reserve_summary.json", "w") as f:
        json.dump({"forecast_file": str(pred), "reserve_config": rcfg,
                   "note": "Demand is SYNTHETIC; capacities simulated; PV power modeled from "
                           "real irradiance; cost coefficients are simulation assumptions.",
                   "results": table.to_dict(orient="records")}, f, indent=2, default=str)
    print("\nSaved: outputs/tables/reserve_summary.csv, outputs/metrics/reserve_summary.json")


if __name__ == "__main__":
    main()
