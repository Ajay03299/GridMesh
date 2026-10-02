"""Phase 10 plots — every figure is drawn from CSVs written by run_experiments.py.

    python -m src.visualization.plots        # re-draw from existing outputs

Style rules (kept deliberately plain): one y-axis per chart, thin marks, hairline grid,
a fixed colour per METHOD across all figures, faulty sites in red with a text label,
healthy sites in grey. Every chart has a CSV twin in outputs/tables/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from scipy.stats import norm  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE, FAULT = "#fcfcfb", "#d03b3b"
METHOD = {   # colour follows the method in every figure (validated categorical order)
    "reliability_fedavg": ("#2a78d6", "Reliability-aware FedAvg (ours)"),
    "fedavg": ("#eb6834", "FedAvg"),
    "reliability_fedavg_event": ("#1baf7a", "Ours + event-aware"),
    "centralized": ("#eda100", "Centralized (pooled data)"),
    "local_only": ("#e87ba4", "Local-only"),
    "smart_persistence": (MUTED, "Smart persistence"),
    "persistence": (AXIS, "Persistence"),
}
FL = ["fedavg", "reliability_fedavg", "reliability_fedavg_event"]
HEALTHY_GREYS = ["#52514e", "#898781", "#b0aea6", "#6b6a66"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "figure.dpi": 110,
})


def _style(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def _save(fig, out, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7.5, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, 1))
    fig.savefig(out / "plots" / name, dpi=160)
    plt.close(fig)


def _read(path):
    return pd.read_csv(path) if Path(path).exists() else None


def _err(df, col):
    return df[f"{col}_std"].fillna(0).to_numpy()


# --------------------------------------------------------------------- FL plots
def plot_convergence(out):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in FL:
        d = _read(out / "logs" / f"exp_normal_{m}_rounds.csv")
        if d is None:
            continue
        c, label = METHOD[m]
        ax.plot(d["round"], np.sqrt(d["val_mse_healthy"]), color=c, lw=1.8, label=label)
        kept = int(d.loc[d["val_reported"].idxmin(), "round"]) - 1   # model the server keeps
        if kept >= 1:
            ax.plot(kept, np.sqrt(d.loc[d["round"] == kept, "val_mse_healthy"].iloc[0]), "o",
                    color=c, ms=8, mec=SURFACE, mew=1.5, zorder=3)
    ax.plot([], [], "o", color=MUTED, ms=8, label="round the server keeps")
    ax.set(xlabel="Federated round", ylabel="Validation RMSE (p.u. of capacity)",
           title="1 · Convergence — global model on every site's validation data")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _style(ax)
    ax.legend()
    _save(fig, out, "01_convergence.png", "First seed. Dots = round kept by the server (lowest "
          "client-reported validation error); later rounds over-fit.")


def plot_actual_vs_pred(out, cfg):
    sp = _read(out / "tables" / "exp_pred_smart_persistence.csv")
    ours = _read(out / "tables" / "exp_pred_reliability_fedavg.csv")
    if sp is None or ours is None:
        return
    def node(df):
        df = df[(df.split == "test") & df.daytime].copy()
        df["time"] = pd.to_datetime(df["time"])
        df["mw_f"], df["mw_a"] = df.capacity_mw * df.forecast_pu, df.capacity_mw * df.actual_pu
        return df.groupby("time")[["mw_f", "mw_a"]].sum()
    a, b = node(ours), node(sp)
    month = cfg.get("plots", {}).get("example_month", 7)
    days = sorted({t.date() for t in a.index if t.month == month})[:3]
    win = a[[t.date() in days for t in a.index]]
    bw = b.loc[win.index]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(win))
    ax.plot(x, win.mw_a, color=INK, lw=1.6, label="Actual (modeled from real irradiance)")
    ax.plot(x, bw.mw_f.to_numpy(), color=MUTED, lw=1.2, label="Smart persistence")
    ax.plot(x, win.mw_f, color=METHOD["reliability_fedavg"][0], lw=1.8,
            label="Ours, +30 min forecast")
    ticks = [i for i, t in enumerate(win.index) if t.hour == 12 and t.minute == 0]
    ax.set_xticks(ticks, [win.index[i].strftime("%d %b") for i in ticks])
    ax.set(ylabel="Node PV power (MW)",
           title="2 · Actual vs forecast at the aggregation node — example test days")
    _style(ax)
    ax.legend(loc="upper left")
    _save(fig, out, "02_actual_vs_predicted.png",
          "Night-time gaps removed. 4 simulated sites, 51 MW. PV power is MODELED from real irradiance.")


def plot_per_site(out):
    d = _read(out / "tables" / "exp_per_site.csv")
    if d is None:
        return
    d = d[d.scenario == "normal"]
    methods = ["smart_persistence", "local_only", "centralized", "fedavg", "reliability_fedavg"]
    g = d.groupby(["site", "method"]).rmse.agg(["mean", "std"]).reset_index()
    sites = sorted(g.site.unique())
    if len(sites) > 12:   # 100-site mode: distribution instead of bars
        fig, ax = plt.subplots(figsize=(8, 4))
        data = [g[g.method == m]["mean"].to_numpy() for m in methods]
        ax.boxplot(data, labels=[METHOD[m][1] for m in methods], vert=False)
        ax.set(xlabel="Per-site test RMSE (p.u.)", title="3 · Per-site error across all sites")
        _style(ax, "x")
        _save(fig, out, "03_per_site_error.png")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        gm = g[g.method == m].set_index("site").reindex(sites)
        ax.bar(np.arange(len(sites)) + (i - (len(methods) - 1) / 2) * w, gm["mean"],
               w * 0.88, yerr=gm["std"].fillna(0), color=METHOD[m][0], label=METHOD[m][1],
               error_kw={"elinewidth": 0.8, "ecolor": INK2, "capsize": 0})
    ax.set_xticks(np.arange(len(sites)), [f"Site {s}" for s in sites])
    lo = g["mean"].min()
    ax.set_ylim(lo * 0.85, g["mean"].max() * 1.04)
    ax.set(ylabel="Test RMSE (p.u. of capacity)", title="3 · Per-site forecast error")
    _style(ax)
    ax.legend(ncol=3, fontsize=8.5, loc="upper left")
    _save(fig, out, "03_per_site_error.png",
          "Mean ± std over seeds. Y-axis does not start at 0 (differences are small).")


def _site_lines(ax, d, value, faulty):
    ends = []
    for i, (site, g) in enumerate(d.groupby("site")):
        bad = site in faulty
        c = FAULT if bad else HEALTHY_GREYS[i % len(HEALTHY_GREYS)]
        ax.plot(g["round"], g[value], color=c, lw=2.0 if bad else 1.3,
                label=f"Site {site}" + (" (faulty)" if bad else ""))
        ends.append((g.iloc[-1][value], site, bad, g.iloc[-1]["round"]))
    # one end label per cluster of lines that finish at (almost) the same value
    span = max(1e-9, d[value].abs().max())
    ends.sort()
    groups = []
    for e in ends:
        if groups and abs(e[0] - groups[-1][-1][0]) < 0.05 * span and e[2] == groups[-1][-1][2]:
            groups[-1].append(e)
        else:
            groups.append([e])
    for grp in groups:
        bad = grp[0][2]
        txt = ", ".join(x[1] for x in grp) + (" faulty" if bad else "")
        y = sum(x[0] for x in grp) / len(grp)
        ax.annotate(txt, (grp[0][3], y), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color=FAULT if bad else INK2, va="center")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _fault_scenario(out, cfg):
    sc = cfg.get("plots", {}).get("fault_scenario", "fault_stale_CD")
    return sc, sc.split("_")[-1]


def plot_trust(out, cfg):
    sc, faulty = _fault_scenario(out, cfg)
    d = _read(out / "logs" / f"exp_{sc}_reliability_fedavg_clients.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    _site_lines(ax, d, "trust", faulty)
    q = cfg["reliability"]["quarantine_below"]
    ax.axhline(q, color=AXIS, lw=0.9)
    ax.text(1, q + 0.02, f"quarantine below {q}", fontsize=8, color=MUTED)
    ax.set(xlabel="Federated round", ylabel="Trust score (0–1)", ylim=(-0.03, 1.08),
           title="4 · Client trust score — reliability-aware FedAvg")
    _style(ax)
    ax.legend(fontsize=8.5, loc="center right")
    _save(fig, out, "04_client_trust.png", f"Scenario: {sc} (simulated sensor fault), first seed.")


def plot_weights(out, cfg):
    sc, faulty = _fault_scenario(out, cfg)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m in zip(axes, ["fedavg", "reliability_fedavg"]):
        d = _read(out / "logs" / f"exp_{sc}_{m}_clients.csv")
        if d is None:
            continue
        _site_lines(ax, d, "weight", faulty)
        ax.set(xlabel="Federated round", title=METHOD[m][1])
        _style(ax)
    axes[0].set_ylabel("Aggregation weight")
    fig.suptitle("5 · Aggregation weights by round", x=0.01, ha="left", fontweight="bold")
    _save(fig, out, "05_aggregation_weights.png",
          f"Scenario: {sc}. Weights are logged by the simulation (normalised to sum to 1).")


def plot_participation(out):
    d = _read(out / "logs" / "exp_dropout_0.25_reliability_fedavg_event_clients.csv")
    if d is None:
        return
    status = {"active": 2, "skipped": 1, "dropped": 0}
    piv = d.pivot(index="site", columns="round", values="status").replace(status)
    drift = d.pivot(index="site", columns="round", values="drift")
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([METHOD["fedavg"][0], GRID, METHOD["reliability_fedavg"][0]])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(piv) + 1.8))
    ax.imshow(piv.to_numpy(float), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    ys, xs = np.where(drift.to_numpy(bool))
    ax.scatter(xs, ys, marker="o", s=26, facecolor=SURFACE, edgecolor=INK, lw=1, zorder=3)
    ax.set_yticks(range(len(piv)), [f"Site {s}" for s in piv.index])
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ax.set_xticks(np.arange(-0.5, len(piv.columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(piv)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    import matplotlib.patches as mp
    handles = [mp.Patch(color=METHOD["reliability_fedavg"][0], label="trained (active)"),
               mp.Patch(color=GRID, label="skipped (16-byte heartbeat)"),
               mp.Patch(color=METHOD["fedavg"][0], label="dropped out"),
               plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK, label="drift detected")]
    ax.legend(handles=handles, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.18))
    ax.set(xlabel="Federated round", title="6 · Client participation — event-aware, 25% dropout")
    _save(fig, out, "06_participation.png")


# ------------------------------------------------------------------- reserve plots
def plot_error_distribution(out):
    d = _read(out / "tables" / "exp_reserve_ts_fixed_20pct.csv")
    if d is None:
        return
    e = d[(d.split == "test") & d.daytime].eval("actual_mw - forecast_mw")
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(e.quantile(0.002), e.quantile(0.998), 70)
    ax.hist(e, bins=bins, density=True, color=METHOD["reliability_fedavg"][0], alpha=0.85,
            rwidth=0.9, label="Measured node error")
    x = np.linspace(bins[0], bins[-1], 300)
    ax.plot(x, norm.pdf(x, e.mean(), e.std()), color=INK, lw=1.4, label="Gaussian, same mean/std")
    q = e.quantile(0.05)
    ax.axvline(q, color=FAULT, lw=1)
    ax.text(q, ax.get_ylim()[1] * 0.92, f" 5th pct {q:.1f} MW\n (Gaussian: "
            f"{e.mean() + norm.ppf(0.05) * e.std():.1f})", color=FAULT, fontsize=8)
    ax.set(xlabel="Forecast error, actual − forecast (MW)", ylabel="Density",
           title="7 · Forecast error distribution at the node")
    _style(ax)
    ax.legend(loc="upper right", fontsize=8.5)
    _save(fig, out, "07_error_distribution.png",
          "Daytime test intervals, reliability-aware forecasts. Peaked + heavy tails vs Gaussian.")


def plot_reserve_time(out, cfg):
    fx = _read(out / "tables" / "exp_reserve_ts_fixed_20pct.csv")
    ns = _read(out / "tables" / f"exp_reserve_ts_nsigma_d{cfg['reserve']['delta']}.csv")
    if fx is None or ns is None:
        return
    fx["time"] = pd.to_datetime(fx["time"])
    month = cfg.get("plots", {}).get("example_month", 7)
    t = fx[(fx.split == "test") & (fx.time.dt.month == month)]
    days = sorted(t.time.dt.date.unique())[:3]
    m = fx.time.dt.date.isin(days) & (fx.split == "test") & fx.daytime
    x = np.arange(m.sum())
    deficit = fx.loc[m, "deficit_mw"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(x, 0, deficit, color=GRID, step="mid", label="Shortfall to cover (forecast − actual)")
    ax.plot(x, fx.loc[m, "reserve_mw"], color=METHOD["fedavg"][0], lw=1.6, label="Fixed 20% reserve")
    ax.plot(x, ns.loc[m.to_numpy(), "reserve_mw"], color=METHOD["reliability_fedavg"][0], lw=1.8,
            label=f"n-sigma reserve (δ = {cfg['reserve']['delta']})")
    miss = ns.loc[m.to_numpy(), "shortfall_mw"].to_numpy() > 1e-9
    ax.scatter(x[miss], deficit[miss], s=14, color=FAULT, zorder=3, label="n-sigma not enough")
    tt = fx.loc[m, "time"].reset_index(drop=True)
    ticks = [i for i, v in enumerate(tt) if v.hour == 12 and v.minute == 0]
    ax.set_xticks(ticks, [tt[i].strftime("%d %b") for i in ticks])
    ax.set_ylim(0, max(deficit.max(), 1) * 1.35)
    ax.set(ylabel="MW", title="8 · Reserve requirement over time — example test days")
    _style(ax)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2)
    _save(fig, out, "08_reserve_over_time.png",
          "Daytime intervals only (night removed). Demand is SYNTHETIC; capacities simulated.")


def _reserve_table(out, cfg):
    r = _read(out / "tables" / "reserve_comparison.csv")
    if r is None:
        return None
    return r[r.forecast == cfg["reserve"]["forecast_method"]].reset_index(drop=True)


def plot_reserve_tradeoff(out, cfg):
    r = _reserve_table(out, cfg)
    if r is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ns = r[r.policy.str.startswith("nsigma")].sort_values("reserve_pct_of_forecast_mean")
    fx = r[r.policy.str.startswith("fixed")]
    c = METHOD["reliability_fedavg"][0]
    ax.errorbar(ns.reserve_pct_of_forecast_mean, ns.ens_pct_of_demand_mean,
                xerr=_err(ns, "reserve_pct_of_forecast"), yerr=_err(ns, "ens_pct_of_demand"),
                color=c, lw=1.8, marker="o", ms=6, elinewidth=0.8, capsize=0,
                label="n-sigma (varying δ)")
    for _, row in ns.iterrows():
        ax.annotate(f"δ={row.policy.split('_d')[1]}", (row.reserve_pct_of_forecast_mean,
                    row.ens_pct_of_demand_mean), xytext=(7, 5), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.errorbar(fx.reserve_pct_of_forecast_mean, fx.ens_pct_of_demand_mean,
                yerr=_err(fx, "ens_pct_of_demand"), color=METHOD["fedavg"][0], marker="s",
                ms=7, ls="", elinewidth=0.8, label="Fixed 20% of forecast")
    ax.set(xlabel="Reserve held (% of forecast renewable energy)",
           ylabel="Energy not served (% of demand)",
           title="9 · Reserve vs shortfall trade-off (down-left is better)")
    _style(ax, "both")
    ax.legend(fontsize=8.5)
    _save(fig, out, "09_reserve_tradeoff.png",
          "Daytime test intervals, mean ± std over seeds. Demand SYNTHETIC; capacities simulated.")


def plot_reserve_bars(out, cfg):
    r = _reserve_table(out, cfg)
    if r is None:
        return
    labels = [p.replace("fixed_20pct", "Fixed 20%").replace("nsigma_d", "n-sigma δ=") for p in r.policy]
    colors = [METHOD["fedavg"][0] if p.startswith("fixed") else METHOD["reliability_fedavg"][0]
              for p in r.policy]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, [("reserve_pct_of_forecast", "Reserve held (% of forecast)"),
                                       ("ens_pct_of_demand", "Energy not served (% of demand)"),
                                       ("availability_pct", "Intervals fully covered (%)")]):
        y = np.arange(len(r))
        ax.barh(y, r[f"{col}_mean"], 0.62, xerr=_err(r, col), color=colors,
                error_kw={"elinewidth": 0.8, "ecolor": INK2})
        for yi, v, e in zip(y, r[f"{col}_mean"], _err(r, col)):
            ax.text(v + e, yi, f"  {v:.2f}" if v < 5 else f"  {v:.1f}", va="center",
                    fontsize=8, color=INK2)
        if col == "availability_pct":
            for yi, p in zip(y, r.policy):
                if p.startswith("nsigma"):
                    tgt = 100 * (1 - float(p.split("_d")[1]))
                    ax.plot([tgt, tgt], [yi - 0.35, yi + 0.35], color=INK, lw=1.4)
            ax.set_xlim(left=min(80, r[f"{col}_mean"].min() - 3))
            ax.text(0.98, -0.2, "| = target (1 − δ)", transform=ax.transAxes, ha="right",
                    fontsize=7.5, color=MUTED)
        ax.set_title(title, fontsize=10)
        _style(ax, "x")
    axes[0].set_yticks(np.arange(len(r)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("10 · Fixed vs n-sigma reserve", x=0.01, ha="left", fontweight="bold")
    _save(fig, out, "10_reserve_policies.png", "Daytime test intervals, mean ± std over seeds.")


def plot_cost(out, cfg):
    r = _reserve_table(out, cfg)
    if r is None:
        return
    rc = cfg["reserve"]
    # cost is linear in energy, so mean cost = coefficient x mean energy (exact)
    g = pd.DataFrame({"cost_reserve": rc["cost_reserve_per_mwh"] * r.reserve_energy_mwh_mean,
                      "cost_shortfall": rc["cost_shortfall_per_mwh"] * r.shortfall_energy_mwh_mean})
    g.index = r.policy
    labels = [p.replace("fixed_20pct", "Fixed 20%").replace("nsigma_d", "n-sigma δ=") for p in g.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(g))
    ax.barh(y, g.cost_reserve, 0.6, color=METHOD["reliability_fedavg"][0], label="Reserve cost")
    ax.barh(y, g.cost_shortfall, 0.6, left=g.cost_reserve + g.values.sum(axis=1).max() * 0.004,
            color=FAULT, label="Shortfall penalty")
    for yi, tot in zip(y, g.sum(axis=1)):
        ax.text(tot * 1.01, yi, f"{tot:,.0f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set(xlabel="Total simulated cost (cost units)",
           title="11 · Cost comparison (coefficients are assumptions)")
    _style(ax, "x")
    ax.set_xlim(0, g.values.sum(axis=1).max() * 1.12)
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    _save(fig, out, "11_cost_comparison.png",
          f"Assumed {rc['cost_reserve_per_mwh']:g} unit/MWh reserve, "
          f"{rc['cost_shortfall_per_mwh']:g} units/MWh not served. Ranking depends on this ratio.")


# -------------------------------------------------------------- robustness plots
def plot_faults(out):
    f = _read(out / "tables" / "fault_scenarios.csv")
    s = _read(out / "tables" / "severity_sweep.csv")
    if f is None or s is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    scen = list(dict.fromkeys(f.scenario))
    w = 0.8 / len(FL)
    for i, m in enumerate(FL):
        g = f[f.method == m].set_index("scenario").reindex(scen)
        ax.barh(np.arange(len(scen)) + (i - 1) * w, g.damage_pct_mean, w * 0.88,
                xerr=g.damage_pct_std.fillna(0), color=METHOD[m][0], label=METHOD[m][1],
                error_kw={"elinewidth": 0.8, "ecolor": INK2})
    ax.axvline(0, color=AXIS, lw=0.9)
    nice = [sc.replace("fault_", "").replace("_", " · ").replace("feature · corruption",
            "feature corruption").replace("target · noise", "target noise") for sc in scen]
    ax.set_yticks(np.arange(len(scen)), [n.rsplit(" · ", 1)[0] + "  [site " +
                  n.rsplit(" · ", 1)[1] + "]" for n in nice])
    ax.invert_yaxis()
    ax.set(xlabel="Healthy-site RMSE increase vs no fault (%)", title="Damage to healthy sites")
    _style(ax, "x")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    ax = axes[1]
    for m in ["fedavg", "reliability_fedavg"]:
        g = s[s.method == m]
        ax.errorbar(g.severity, g.healthy_rmse_mean, yerr=_err(g, "healthy_rmse"),
                    color=METHOD[m][0], lw=1.8, marker="o", ms=5, elinewidth=0.8, label=METHOD[m][1])
    ax.set(xlabel="Fault severity (0 = clean)", ylabel="Healthy-site RMSE (p.u.)",
           title="Stale-sensor fault on site D")
    _style(ax)
    ax.legend(fontsize=8)
    fig.suptitle("12 · Faulty-client experiment", x=0.01, ha="left", fontweight="bold")
    _save(fig, out, "12_faulty_client.png",
          "Simulated sensor/meter faults present for the whole training run. Mean ± std over seeds.")


def plot_dropout(out):
    d = _read(out / "tables" / "dropout.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in FL:
        g = d[d.method == m].sort_values("dropout")
        ax.errorbar(g.dropout * 100, g.global_rmse_mean, yerr=_err(g, "global_rmse"),
                    color=METHOD[m][0], lw=1.8, marker="o", ms=5, elinewidth=0.8, label=METHOD[m][1])
    ax.set(xlabel="Client dropout rate (%)", ylabel="Test RMSE (p.u. of capacity)",
           title="13 · Client dropout experiment")
    _style(ax)
    ax.legend(fontsize=8.5)
    _save(fig, out, "13_dropout.png", "Each selected client fails at random with this probability. "
          "Mean ± std over seeds.")


def plot_scaling(out):
    d = _read(out / "tables" / "comm_scaling.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for m in ["fedavg", "reliability_fedavg_event"]:
        g = d[d.method == m].sort_values("n_sites")
        axes[0].plot(g.n_sites, g.comm_mb_per_round, color=METHOD[m][0], lw=1.8, marker="o",
                     ms=5, label=METHOD[m][1])
        axes[1].plot(g.n_sites, g.global_rmse, color=METHOD[m][0], lw=1.8, marker="o", ms=5,
                     label=METHOD[m][1])
        last = g.iloc[-1]
        axes[0].annotate(f"{last.comm_mb_per_round:.1f} MB", (last.n_sites, last.comm_mb_per_round),
                         xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=8,
                         color=INK2)
    axes[0].set(xlabel="Number of sites", ylabel="Traffic per round (MB)",
                title="Communication per round")
    axes[1].set(xlabel="Number of sites", ylabel="Test RMSE (p.u.)", title="Accuracy",
                ylim=(0, d.global_rmse.max() * 1.3))
    axes[1].text(0.02, 0.06, f"spread across all runs: {d.global_rmse.max() - d.global_rmse.min():.4f} p.u.",
                 transform=axes[1].transAxes, fontsize=8, color=MUTED)
    for ax in axes:
        _style(ax)
    axes[0].legend(fontsize=8.5)
    fig.suptitle("14 · Scaling from 4 to 500 logical clients", x=0.01, ha="left",
                 fontweight="bold")
    _save(fig, out, "14_comm_scaling.png",
          "Cap 20. Three model transfers + metadata. At 50+ clients, compact datasets repeat; accuracy is illustrative.")


def make_all(out, cfg):
    out = Path(out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    steps = [plot_convergence, lambda o: plot_actual_vs_pred(o, cfg), plot_per_site,
             lambda o: plot_trust(o, cfg), lambda o: plot_weights(o, cfg), plot_participation,
             plot_error_distribution, lambda o: plot_reserve_time(o, cfg),
             lambda o: plot_reserve_tradeoff(o, cfg), lambda o: plot_reserve_bars(o, cfg),
             lambda o: plot_cost(o, cfg), plot_faults, plot_dropout, plot_scaling]
    for fn in steps:
        fn(out)
    n = len(list((out / "plots").glob("*.png")))
    print(f"Saved {n} plots to {out / 'plots'}")


if __name__ == "__main__":
    from src.data.adapter import load_config
    c = load_config()
    make_all(Path(c["paths"]["outputs"]), c)
