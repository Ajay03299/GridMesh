"""Phase 10 — the full experiment matrix, repeated over several seeds.

    python run_experiments.py            # full matrix (a few minutes on a MacBook)
    python run_experiments.py --quick    # 1 seed, fewer scenarios (smoke test, ~1 min)

Scenarios (all in config.yaml -> experiments):
  normal    : every method (baselines + 3 FL variants)
  faults    : FL methods with 1 or 2 degraded sites (from faults.start_round)
  severity  : FedAvg vs reliability-aware as one fault gets stronger
  dropout   : FL methods with random client dropout
  reserve   : fixed vs n-sigma reserve on each seed's forecasts

Writes outputs/tables/*.csv, outputs/metrics/summary.json and the plots in outputs/plots/.
Every number comes from the runs below; nothing is hard-coded.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd

from run_simulation import BASELINES, FL, run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate, predictions_frame
from src.models.forecasting_model import set_seed
from src.reserve.simulator import load_node, run_policies

RESERVE_SOURCES = ["smart_persistence", "fedavg", "reliability_fedavg"]


def site_idx(letters):
    return [ord(s.upper()) - ord("A") for s in letters]


class Runner:
    """Runs one scenario for one seed; caches results so repeated scenarios are free."""

    def __init__(self, cfg, out, n_clients):
        self.cfg, self.out, self.n = cfg, out, n_clients
        self.rows, self.site_rows, self.reserve_rows, self.cache = [], [], [], {}

    def run(self, scenario, seed, methods, fault_type=None, sites=(), severity=1.0,
            dropout=0.0, keep_logs=False, reserve=False):
        key = (fault_type, tuple(sites), severity, dropout, seed)
        todo = [m for m in methods if (key, m) not in self.cache]
        if todo:
            cfg = copy.deepcopy(self.cfg)
            cfg["federated"]["dropout_rate"] = dropout
            set_seed(seed)
            faulty = site_idx(sites)
            clients, _ = prepare_clients(cfg, self.n, seed, faulty, fault_type, severity)
            for m in todo:
                res = run_method(m, clients, cfg, seed, faulty, verbose=False)
                s = evaluate(res, clients)
                per_site = s.pop("per_site")
                self.cache[(key, m)] = s
                for site, v in per_site.items():
                    self.site_rows.append({"scenario": scenario, "seed": seed, "method": m,
                                           "site": site, "rmse": v["rmse"], "mae": v["mae"],
                                           "n": v["n"]})
                if keep_logs and m in FL:
                    pd.DataFrame(res.history).to_csv(
                        self.out / "logs" / f"exp_{scenario}_{m}_clients.csv", index=False)
                    res.extra["rounds"].to_csv(
                        self.out / "logs" / f"exp_{scenario}_{m}_rounds.csv", index=False)
                if reserve and m in RESERVE_SOURCES:
                    pred = predictions_frame(res, clients)
                    if keep_logs:
                        pred.to_csv(self.out / "tables" / f"exp_pred_{m}.csv", index=False)
                    for ts, summ in run_policies(load_node(pred), cfg, seed):
                        self.reserve_rows.append({"seed": seed, "forecast": m, **summ})
                        if keep_logs and m == cfg["reserve"]["forecast_method"]:
                            ts.to_csv(self.out / "tables" / f"exp_reserve_ts_{summ['policy']}.csv",
                                      index=False)
        for m in methods:
            self.rows.append({"scenario": scenario, "seed": seed, "fault_type": fault_type,
                              "n_faulty": len(sites), "severity": severity if sites else 0.0,
                              "dropout": dropout, **self.cache[(key, m)]})


def subset_rmse(site_df, scenario, seed, method, sites):
    """Pooled RMSE of one run restricted to `sites` (sqrt of n-weighted mean squared error)."""
    d = site_df[(site_df.scenario == scenario) & (site_df.seed == seed)
                & (site_df.method == method) & site_df.site.isin(sites)]
    return float(((d.n * d.rmse ** 2).sum() / d.n.sum()) ** 0.5)


def mean_std(df, by, cols):
    g = df.groupby(by, sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{c}_{stat}" for c, stat in g.columns]
    return g.reset_index()


def pm(m, s, digits=4):
    return f"{m:.{digits}f} ± {0 if pd.isna(s) else s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="1 seed, fewer scenarios")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ex = cfg["experiments"]
    seeds = ex["seeds"][:1] if args.quick else ex["seeds"]
    faults = ex["fault_scenarios"][:2] if args.quick else ex["fault_scenarios"]
    sweep = ex["severity_sweep"]
    severities = [s for s in sweep["severities"] if s != 1][:1] if args.quick else sweep["severities"]
    dropouts = ex["dropout_rates"][:1] if args.quick else ex["dropout_rates"]
    out = Path(cfg["paths"]["outputs"])
    for sub in ("metrics", "tables", "logs", "plots"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    R = Runner(cfg, out, args.clients)
    t_all = time.time()
    for i, seed in enumerate(seeds):
        first = i == 0
        jobs = [("normal", dict(methods=BASELINES + FL, keep_logs=first, reserve=True))]
        for f in faults:
            name = f"fault_{f['type']}_{''.join(f['sites'])}"
            jobs.append((name, dict(methods=FL, fault_type=f["type"], sites=f["sites"],
                                    severity=f["severity"], keep_logs=first)))
        for sev in severities:
            jobs.append((f"sweep_{sweep['type']}_s{sev:g}",
                         dict(methods=["fedavg", "reliability_fedavg"], fault_type=sweep["type"],
                              sites=sweep["sites"], severity=sev)))
        for d in dropouts:
            jobs.append((f"dropout_{d:g}", dict(methods=FL, dropout=d, keep_logs=first)))
        for name, kw in jobs:
            t = time.time()
            R.run(name, seed, **kw)
            print(f"  seed {seed} | {name:<32} {time.time() - t:5.1f}s")

    runs = pd.DataFrame(R.rows)
    runs.to_csv(out / "tables" / "exp_all_runs.csv", index=False)
    sites_df = pd.DataFrame(R.site_rows)
    sites_df.to_csv(out / "tables" / "exp_per_site.csv", index=False)
    all_sites = sorted(sites_df.site.unique())
    normal = runs[runs.scenario == "normal"]

    # ---- 1. main comparison table
    main_t = mean_std(normal, "method", ["global_rmse", "global_mae", "worst_site_rmse",
                                         "rounds_to_converge", "total_comm_mb",
                                         "participation_rate"])
    clean = normal.set_index(["method", "seed"])
    fault_runs = runs[runs.scenario.str.startswith("fault_")].copy()
    # damage = healthy-site RMSE vs the SAME sites in the no-fault run (same seed, same method)
    fault_runs["clean_healthy_rmse"] = [
        subset_rmse(sites_df, "normal", r.seed, r.method,
                    [x for x in all_sites if x not in str(r.faulty_sites).split(",")])
        for r in fault_runs.itertuples()]
    fault_runs["damage_pct"] = 100 * (fault_runs.healthy_rmse / fault_runs.clean_healthy_rmse - 1)
    drop_runs = runs[runs.scenario.str.startswith("dropout_")].copy()
    drop_runs["degradation_pct"] = [
        100 * (r.global_rmse / clean.loc[(r.method, r.seed), "global_rmse"] - 1)
        for r in drop_runs.itertuples()]
    main_t = main_t.merge(mean_std(fault_runs, "method", ["damage_pct"]), on="method", how="left")
    main_t = main_t.merge(mean_std(drop_runs, "method", ["degradation_pct"]), on="method", how="left")
    main_t.to_csv(out / "tables" / "comparison_main.csv", index=False)

    # ---- 2. fault scenarios / severity / dropout
    fault_t = mean_std(fault_runs, ["scenario", "method"],
                       ["clean_healthy_rmse", "healthy_rmse", "damage_pct"])
    fault_t.to_csv(out / "tables" / "fault_scenarios.csv", index=False)
    sweep_runs = pd.concat([
        normal[normal.method.isin(["fedavg", "reliability_fedavg"])],
        runs[(runs.fault_type == sweep["type"]) & (runs.n_faulty == len(sweep["sites"]))
             & runs.method.isin(["fedavg", "reliability_fedavg"])]])
    sweep_t = mean_std(sweep_runs, ["severity", "method"], ["healthy_rmse"]).sort_values("severity")
    sweep_t.to_csv(out / "tables" / "severity_sweep.csv", index=False)
    drop_t = mean_std(pd.concat([normal[normal.method.isin(FL)], drop_runs]),
                      ["dropout", "method"], ["global_rmse", "total_comm_mb"])
    drop_t.to_csv(out / "tables" / "dropout.csv", index=False)

    # ---- 3. reserve
    res = pd.DataFrame(R.reserve_rows)
    res.to_csv(out / "tables" / "exp_reserve_runs.csv", index=False)
    res_t = mean_std(res, ["forecast", "policy"], [
        "reserve_energy_mwh", "shortfall_energy_mwh", "reserve_pct_of_forecast",
        "ens_pct_of_demand", "availability_pct", "total_cost", "cost_reserve", "cost_shortfall"])
    res_t.to_csv(out / "tables" / "reserve_comparison.csv", index=False)

    # ---- print
    n_seeds = len(seeds)
    print(f"\n=== MAIN COMPARISON (test, daytime, p.u.; mean ± std over {n_seeds} seed(s)) ===")
    show = pd.DataFrame({
        "method": main_t["method"],
        "RMSE": [pm(a, b) for a, b in zip(main_t.global_rmse_mean, main_t.global_rmse_std)],
        "MAE": [pm(a, b) for a, b in zip(main_t.global_mae_mean, main_t.global_mae_std)],
        "worst-site RMSE": [pm(a, b) for a, b in zip(main_t.worst_site_rmse_mean,
                                                     main_t.worst_site_rmse_std)],
        "comm MB": main_t.total_comm_mb_mean.round(2),
        "fault damage %": [pm(a, b, 1) if pd.notna(a) else "-" for a, b in
                           zip(main_t.damage_pct_mean, main_t.damage_pct_std)],
        "dropout degr. %": [pm(a, b, 1) if pd.notna(a) else "-" for a, b in
                            zip(main_t.degradation_pct_mean, main_t.degradation_pct_std)],
    })
    print(show.to_string(index=False))
    print("\n=== FAULT SCENARIOS: healthy-site RMSE increase vs no fault (%) ===")
    piv = fault_t.pivot(index="scenario", columns="method", values="damage_pct_mean")
    print(piv[[m for m in FL if m in piv]].round(2).to_string())
    print("\n=== SEVERITY SWEEP: healthy-site RMSE ===")
    print(sweep_t.pivot(index="severity", columns="method",
                        values="healthy_rmse_mean").round(4).to_string())
    print("\n=== RESERVE (daytime test; costs are SIMULATION ASSUMPTIONS; demand SYNTHETIC) ===")
    rshow = res_t[res_t.forecast == cfg["reserve"]["forecast_method"]]
    print(rshow[["policy", "reserve_energy_mwh_mean", "shortfall_energy_mwh_mean",
                 "availability_pct_mean", "total_cost_mean"]].round(1).to_string(index=False))

    # ---- summary.json (headline metrics, all measured)
    def rec(df, key):
        return {r[key]: {k: v for k, v in r.items() if k != key} for r in df.to_dict("records")}
    summary = {
        "seeds": seeds, "n_clients": args.clients,
        "provenance": {
            "weather": "REAL (NSRDB-format 10-min file, year assumed 2019)",
            "pv_power": "MODELED from real GHI + temperature (simplified PVWatts)",
            "sites": "SIMULATED heterogeneity from one real weather record",
            "faults": "SIMULATED sensor/meter degradation",
            "demand": "SYNTHETIC", "costs": "SIMULATION ASSUMPTIONS",
            "privacy": "raw data stays local; no secure aggregation / DP implemented"},
        "forecasting": rec(main_t, "method"),
        "fault_scenarios": fault_t.to_dict("records"),
        "severity_sweep": sweep_t.to_dict("records"),
        "dropout": drop_t.to_dict("records"),
        "reserve": res_t.to_dict("records"),
        "config": cfg,
        "runtime_s": round(time.time() - t_all, 1),
    }
    with open(out / "metrics" / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved tables to outputs/tables/, summary to outputs/metrics/summary.json "
          f"({summary['runtime_s']}s)")

    if not args.no_plots:
        from src.visualization.plots import make_all
        make_all(out, cfg)
    from src.evaluation.report import write_results_md
    write_results_md(out, cfg)


if __name__ == "__main__":
    main()
