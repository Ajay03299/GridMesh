"""Write outputs/RESULTS.md from the CSVs produced by run_experiments.py.

    python -m src.evaluation.report

Every number in RESULTS.md is read from outputs/tables/*.csv — nothing is typed by hand.
"""
import json
from pathlib import Path

import pandas as pd

NAMES = {
    "persistence": "Persistence", "smart_persistence": "Smart persistence",
    "local_only": "Local-only", "centralized": "Centralized (pooled data)",
    "fedavg": "FedAvg", "reliability_fedavg": "**Reliability-aware FedAvg (ours)**",
    "reliability_fedavg_event": "**Ours + event-aware participation**",
}


def _pm(m, s, d=4):
    if pd.isna(m):
        return "–"
    return f"{m:.{d}f} ± {0 if pd.isna(s) else s:.{d}f}"


def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def win_counts(runs):
    """Paired comparison: in how many fault runs (scenario x seed) does each method beat FedAvg
    on healthy-site RMSE? Simple, assumption-free evidence to go with the means."""
    d = runs[runs.scenario.str.startswith("fault_")]
    piv = d.pivot_table(index=["scenario", "seed"], columns="method", values="healthy_rmse")
    lines = []
    for m in ["reliability_fedavg", "reliability_fedavg_event"]:
        if m in piv and "fedavg" in piv:
            pair = piv[[m, "fedavg"]].dropna()
            wins = int((pair[m] < pair["fedavg"]).sum())
            lines.append(f"{NAMES[m].strip('*')}: lower healthy-site RMSE than FedAvg in "
                         f"**{wins} of {len(pair)}** fault runs (scenario × seed).")
    return lines


def _win_line(t):
    f = t / "exp_all_runs.csv"
    return "\n\n".join(win_counts(pd.read_csv(f))) if f.exists() else ""


def _scaling_section(t):
    f = t / "comm_scaling.csv"
    if not f.exists():
        return ""
    d = pd.read_csv(f)
    show = pd.DataFrame({
        "Sites": d.n_sites, "Method": d.method.map(NAMES).str.strip("*"),
        "MB per round": d.comm_mb_per_round.round(2),
        "Participation (%)": (100 * d.participation_rate).round(0).astype(int),
        "Test RMSE": d.global_rmse.round(4)})
    return ("\n## 6. Scaling 4 → 100 sites (scale simulation, 1 seed, "
            f"{int(d.rounds.iloc[0])} rounds)\n" + _md(show) + "\n")


def write_results_md(out, cfg):
    out = Path(out)
    t = out / "tables"
    main = pd.read_csv(t / "comparison_main.csv")
    fault = pd.read_csv(t / "fault_scenarios.csv")
    sweep = pd.read_csv(t / "severity_sweep.csv")
    drop = pd.read_csv(t / "dropout.csv")
    res = pd.read_csv(t / "reserve_comparison.csv")
    summ = json.loads((out / "metrics" / "summary.json").read_text())
    seeds = summ["seeds"]

    m = pd.DataFrame({
        "Method": main.method.map(NAMES),
        "RMSE": [_pm(a, b) for a, b in zip(main.global_rmse_mean, main.global_rmse_std)],
        "MAE": [_pm(a, b) for a, b in zip(main.global_mae_mean, main.global_mae_std)],
        "Worst-site RMSE": [_pm(a, b) for a, b in zip(main.worst_site_rmse_mean,
                                                      main.worst_site_rmse_std)],
        "Comm. (MB)": [f"{v:.2f}" if pd.notna(v) else "–" for v in main.total_comm_mb_mean],
        "Fault damage (%)": [_pm(a, b, 1) for a, b in zip(main.damage_pct_mean, main.damage_pct_std)],
        "Dropout degradation (%)": [_pm(a, b, 1) for a, b in
                                    zip(main.degradation_pct_mean, main.degradation_pct_std)],
    })
    fp = fault.pivot(index="scenario", columns="method", values="damage_pct_mean")
    fs = fault.pivot(index="scenario", columns="method", values="damage_pct_std")
    fl = [c for c in ["fedavg", "reliability_fedavg", "reliability_fedavg_event"] if c in fp]
    f = pd.DataFrame({"Scenario": [s.replace("fault_", "").replace("_", " ") for s in fp.index],
                      **{NAMES[c].strip("*"): [_pm(a, b, 1) for a, b in zip(fp[c], fs[c])]
                         for c in fl}})
    runs_file = t / "exp_all_runs.csv"
    if runs_file.exists():   # paired wins per fault type (same scenario, same seed)
        pr = pd.read_csv(runs_file)
        pr = pr[pr.scenario.str.startswith("fault_")].pivot_table(
            index=["scenario", "seed"], columns="method", values="healthy_rmse")
        if "reliability_fedavg" in pr and "fedavg" in pr:
            w = (pr["reliability_fedavg"] < pr["fedavg"]).groupby(level=0)
            f["Ours beats FedAvg (runs)"] = [f"{int(w.sum()[s])} / {int(w.count()[s])}"
                                             for s in fp.index]
    sw = sweep.pivot(index="severity", columns="method", values="healthy_rmse_mean").reset_index()
    sw.columns = ["Severity"] + [NAMES[c].strip("*") for c in sw.columns[1:]]
    sw = sw.round(4)
    dp = drop.pivot(index="dropout", columns="method", values="global_rmse_mean").reset_index()
    dp.columns = ["Dropout rate"] + [NAMES[c].strip("*") for c in dp.columns[1:]]
    dp = dp.round(4)
    rr = res[res.forecast == cfg["reserve"]["forecast_method"]]
    r = pd.DataFrame({
        "Policy": rr.policy.str.replace("fixed_20pct", "Fixed 20%").str.replace("nsigma_d", "n-sigma δ="),
        "Reserve (% of forecast)": [_pm(a, b, 1) for a, b in
                                    zip(rr.reserve_pct_of_forecast_mean, rr.reserve_pct_of_forecast_std)],
        "Not served (% of demand)": [_pm(a, b, 2) for a, b in
                                     zip(rr.ens_pct_of_demand_mean, rr.ens_pct_of_demand_std)],
        "Reserve (MWh)": rr.reserve_energy_mwh_mean.round(0).astype(int),
        "Not served (MWh)": rr.shortfall_energy_mwh_mean.round(0).astype(int),
        "Intervals covered (%)": rr.availability_pct_mean.round(1),
        "Target (%)": [f"{100 * (1 - float(p.split('_d')[1])):.0f}" if "nsigma" in p else "–"
                       for p in rr.policy],
        "Total cost": rr.total_cost_mean.round(0).astype(int),
    })
    rc = cfg["reserve"]
    plots = sorted(p.name for p in (out / "plots").glob("*.png"))
    text = f"""# GridMesh — measured results

Auto-generated by `run_experiments.py` (`python -m src.evaluation.report` to rebuild).
{len(seeds)} seeds ({", ".join(map(str, seeds))}), {summ['n_clients']} sites, test = daytime
intervals of the last 15% of days in every month, errors in p.u. of installed capacity
(0.05 = 5% of capacity). Mean ± std over seeds.

**What is real and what is simulated:** weather/irradiance REAL · PV power MODELED from real
irradiance · sites and faults SIMULATED · demand SYNTHETIC · costs ASSUMED.
Raw data stays at each site; no secure aggregation or differential privacy is implemented.

## 1. Forecasting (no faults)
{_md(m)}

*Fault damage* = healthy-site RMSE increase vs. the same sites without faults, averaged over all
fault scenarios. *Dropout degradation* = RMSE increase vs. no dropout, averaged over dropout rates.

## 2. Faulty-client scenarios — healthy-site RMSE increase (%)
{_md(f)}

Faults are simulated sensor/meter problems present for the whole training run
(`D` = one site of 4, `CD` = two sites of 4).

{_win_line(t)}

## 3. Fault severity sweep — healthy-site RMSE (stale sensor on site D)
{_md(sw)}

## 4. Client dropout — test RMSE
{_md(dp)}

## 5. Reserve at one aggregation node (forecasts: {NAMES[rc['forecast_method']].strip('*')})
{_md(r)}

n-sigma reserve: `r = max(α·σ_e − μ_e, 0)`, `α = Φ⁻¹(1−δ)`, causal 6-hour rolling error statistics
(inspired by Khaing, Kannan & Rao, *Clean Energy* 2026). Costs: {rc['cost_reserve_per_mwh']:g} unit
per MWh of reserve, {rc['cost_shortfall_per_mwh']:g} units per MWh not served — **simulation
assumptions**. Demand is **synthetic**.

""" + _scaling_section(t) + """
## Plots
""" + "\n".join(f"![{p}](plots/{p})" for p in plots) + "\n"
    (out / "RESULTS.md").write_text(text)
    print(f"Saved {out / 'RESULTS.md'}")


if __name__ == "__main__":
    from src.data.adapter import load_config
    c = load_config()
    write_results_md(c["paths"]["outputs"], c)
