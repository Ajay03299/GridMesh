"""GridMesh demo dashboard (a demonstration layer, not the research contribution).

    python build_dashboard.py        # once: pre-computes the 4 scenarios
    streamlit run dashboard.py       # opens in the browser
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.data.adapter import load_config  # noqa: E402
from src.visualization.dashboard_theme import (apply_theme, evidence_chart, palette,
                                               render_figure, styled_table)  # noqa: E402
from src.visualization.plots import _style  # noqa: E402

OURS = "reliability_fedavg_event"
LABELS = {"normal": "NORMAL", "faulty": "FAULTY CLIENT", "dropout": "CLIENT DROPOUT",
          "shift": "WEATHER / REGIME SHIFT"}
st.set_page_config(page_title="GridMesh", layout="wide")
dark_mode = st.sidebar.toggle("Dark mode", value=True, key="dark_mode")
colors = palette(dark_mode)
apply_theme(colors, dark_mode)
INK, MUTED, GRID, AXIS = colors["ink"], colors["muted"], colors["grid"], colors["grid"]
SURFACE, FAULT = colors["surface"], colors["fault"]
BLUE, ORANGE, HEALTHY_GREYS = colors["blue"], colors["orange"], colors["healthy"]
cfg = load_config()
ROOT = Path(cfg["paths"]["outputs"]) / "dashboard"
if not (ROOT / "normal" / "kpi.json").exists():
    st.error("No dashboard data yet. Run:  python build_dashboard.py")
    st.stop()


@st.cache_data
def load(name, revision):
    d = ROOT / name
    node = pd.read_csv(d / "node.csv", parse_dates=["time"])
    return (node, pd.read_csv(d / f"clients_{OURS}.csv"), pd.read_csv(d / "clients_fedavg.csv"),
            pd.read_csv(d / "clients_reliability_fedavg.csv"),
            pd.read_csv(d / "sites.csv"), json.loads((d / "kpi.json").read_text()))


def site_health(cl, sites):
    """Latest trust / quality per site -> healthy / degraded / quarantined (transparent rule)."""
    last = cl.sort_values("round").groupby("site").agg(
        trust=("trust", lambda s: s.dropna().iloc[-1] if s.notna().any() else 1.0),
        quality=("quality", lambda s: s.dropna().iloc[-1] if s.notna().any() else 1.0),
        quarantined=("quarantined", "last"))
    last = last.join(sites.set_index("site"))
    last["state"] = np.where(last.quarantined, "quarantined",
                             np.where((last.trust < 0.8) | (last.quality < 0.9), "degraded", "healthy"))
    return last


def fig(w=6, h=3):
    f, ax = plt.subplots(figsize=(w, h))
    return f, ax


def story(scen, kpi):
    """One plain-language sentence per scenario, built only from the measured numbers."""
    fa, ours, sp, r = kpi["fedavg"], kpi[OURS], kpi["smart_persistence"], kpi["reserve"]
    ns, fx = r[f"guarded_nsigma_d{kpi['delta']}"], r[[p for p in r if p.startswith("fixed")][0]]
    d_err = 100 * (ours["healthy_rmse"] / fa["healthy_rmse"] - 1)
    d_comm = 100 * (1 - ours["total_comm_mb"] / fa["total_comm_mb"])
    d_ens = 100 * (1 - ns["ens_pct_of_demand"] / fx["ens_pct_of_demand"])
    word = "lower" if d_err < 0 else "higher"
    if scen == "faulty":
        return (f"Sites {' and '.join(kpi['faulty_sites'])} have frozen sensors. GridMesh notices "
                f"(their trust drops and their weight falls to almost zero), so forecast error at "
                f"the healthy sites is {abs(d_err):.0f}% {word} than standard federated learning.")
    if scen == "dropout":
        return (f"A quarter of the sites go silent each round. Training carries on with whoever "
                f"is online: healthy-site error {ours['healthy_rmse']:.4f} vs "
                f"{fa['healthy_rmse']:.4f} for standard federated learning.")
    if scen == "shift":
        verdict = ("simple smart persistence wins" if sp["global_rmse"] < ours["global_rmse"]
                   else "GridMesh still beats smart persistence")
        return (f"Trained on January–September, tested on November–December, a season the model "
                f"never saw: {verdict} ({sp['global_rmse']:.4f} vs {ours['global_rmse']:.4f}). "
                f"This is the honest limit that motivates drift handling.")
    ens_word = "less" if d_ens >= 0 else "more"
    return (f"All sites are healthy. GridMesh and standard federated learning have errors "
            f"({ours['healthy_rmse']:.4f} vs {fa['healthy_rmse']:.4f}) while sending {d_comm:.0f}% "
            f"less data, and its guarded forecast-aware reserve leaves {abs(d_ens):.0f}% {ens_word} energy unserved "
            f"than a fixed 20% reserve.")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("GridMesh")
st.sidebar.caption("Reliability-aware federated renewable forecasting")
scen = st.sidebar.radio("Scenario", list(LABELS), format_func=LABELS.get)
replay_revision = tuple((ROOT / scen / filename).stat().st_mtime_ns for filename in
                       ("kpi.json", "node.csv", f"clients_{OURS}.csv"))
node, cl, cl_fa, cl_rel, sites, kpi = load(scen, replay_revision)
day_df = node[node.daytime].copy()
daily_err = day_df.assign(e=(day_df.actual_mw - day_df.forecast_mw).abs()).groupby(
    day_df.time.dt.date).e.sum()
days = list(daily_err.index)
day = st.sidebar.selectbox("Test day", days, index=days.index(daily_err.idxmax()),
                           format_func=lambda d: d.strftime("%d %b %Y"),
                           help="Default: the test day with the largest forecast errors")
today = day_df[day_df.time.dt.date == day].reset_index(drop=True)
i_now = st.sidebar.select_slider("Time of day", options=list(range(len(today))),
                                 value=len(today) // 2,
                                 format_func=lambda i: today.time[i].strftime("%H:%M"))
now = today.iloc[i_now]
st.sidebar.markdown("---")
st.sidebar.caption("Weather/irradiance: REAL · PV power: MODELED from irradiance · sites & faults: "
                   "SIMULATED · demand: SYNTHETIC · costs: ASSUMED. Raw data stays at each site; "
                   "no secure aggregation implemented.")

# ------------------------------------------------------------------ top KPIs
st.subheader(f"{LABELS[scen]} — {kpi['title']}")
st.info(story(scen, kpi))
health = site_health(cl, sites)
last_round = cl[cl["round"] == cl["round"].max()]
online = int((last_round.status != "dropped").sum())
trained = int((last_round.status == "active").sum())
risk = 100 * health.loc[health.state != "healthy", "capacity_mw"].sum() / health.capacity_mw.sum()
risk_label = "LOW" if risk < 10 else "MEDIUM" if risk < 30 else "HIGH"
k = st.columns(5)
k[0].metric("Sites online", f"{online} / {len(sites)}")
skipped = int((last_round.status == "skipped").sum())
k[0].caption(f"last round: {trained} trained · {skipped} heartbeat only · "
             f"{len(sites) - online} dropped")
k[1].metric("Forecast, +30 min", f"{now.forecast_mw:.1f} MW")
k[1].caption(f"actual {now.actual_mw:.1f} MW · {now.capacity_mw:.0f} MW installed")
k[2].metric("Uncertainty (σ)", f"{now.sigma_e:.2f} MW")
k[2].caption("std of the last 6 h of forecast errors")
k[3].metric("Reliability risk", risk_label)
k[3].caption(f"{risk:.0f}% of capacity at degraded sites")
k[4].metric("Backup recommendation", f"{now.reserve_mw:.1f} MW",
            f"{now.reserve_mw - now.reserve_fixed_mw:+.1f} MW vs fixed 20%", delta_color="inverse")
k[4].caption(f"expected gap + n-sigma margin, δ = {kpi['delta']}")

# ------------------------------------------------------------------ middle
x = today.time
c1, c2 = st.columns(2)
with c1:
    f, ax = fig()
    ax.plot(x, today.actual_mw, color=INK, lw=1.5, label="actual")
    ax.plot(x, today.forecast_sp_mw, color=MUTED, lw=1, label="smart persistence")
    ax.plot(x, today.forecast_mw, color=BLUE, lw=1.8, label="GridMesh forecast")
    ax.axvline(now.time, color=AXIS, lw=1)
    ax.set(ylabel="MW", title="Predicted vs actual — aggregation node")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    _style(ax)
    ax.legend(fontsize=7, loc="upper left")
    render_figure(f, colors)
with c2:
    f, ax = fig()
    ax.fill_between(x, 0, today.deficit_mw, color=GRID, step="mid", label="realised net deficit")
    ax.plot(x, today.expected_gap_mw, color=MUTED, lw=1.2, label="expected gap")
    ax.plot(x, today.reserve_fixed_mw, color=ORANGE, lw=1.5, label="fixed-margin schedule")
    ax.plot(x, today.reserve_mw, color=BLUE, lw=1.8,
            label=f"constrained n-sigma schedule (δ={kpi['delta']})")
    miss = today.shortfall_mw > 1e-9
    ax.scatter(x[miss], today.deficit_mw[miss], s=12, color=FAULT, zorder=3, label="not covered")
    ax.axvline(now.time, color=AXIS, lw=1)
    infeasible = today.planned_gap_mw > 1e-9
    ax.scatter(x[infeasible], today.required_backup_mw[infeasible], s=20, marker="x",
               color=FAULT, zorder=4, label="capacity warning")
    ax.set(ylabel="MW", title="Expected gap, uncertainty and backup schedule")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    _style(ax)
    ax.legend(fontsize=7, loc="upper left")
    render_figure(f, colors)

r, fa, ours = kpi["reserve"], kpi["fedavg"], kpi[OURS]
ns = r[f"guarded_nsigma_d{kpi['delta']}"]
fx = r[[p for p in r if p.startswith("fixed")][0]]
s1, s2, s3, s4 = st.columns(4)
s1.metric("Healthy-site RMSE", f"{ours['healthy_rmse']:.4f}",
          f"{100 * (ours['healthy_rmse'] / fa['healthy_rmse'] - 1):+.1f}% vs FedAvg",
          delta_color="inverse")
s1.caption(f"GridMesh, whole test period · FedAvg {fa['healthy_rmse']:.4f}")
s2.metric("Communication", f"{ours['total_comm_mb']:.2f} MB",
          f"{100 * (ours['total_comm_mb'] / fa['total_comm_mb'] - 1):+.0f}% vs FedAvg",
          delta_color="inverse")
s2.caption("20 rounds, model updates + heartbeats")
s3.metric("Energy not served", f"{ns['ens_pct_of_demand']:.2f}%",
          f"{ns['ens_pct_of_demand'] - fx['ens_pct_of_demand']:+.2f} pts vs fixed 20%",
          delta_color="inverse")
s3.caption("% of (synthetic) demand, whole test period")
s4.metric("Intervals fully covered", f"{ns['availability_pct']:.1f}%")
s4.caption(f"fixed 20% covers {fx['availability_pct']:.1f}% · margin target is separate from service availability")

# ------------------------------------------------------------------ bottom
b1, b2, b3 = st.columns(3)
faulty = set(kpi["faulty_sites"])
with b1:
    f, ax = fig(5, 3.2)
    for j, (site, g) in enumerate(cl.groupby("site")):
        bad = site in faulty
        ax.plot(g["round"], g["trust"].ffill(), color=FAULT if bad else HEALTHY_GREYS[j % 4],
                lw=2 if bad else 1.2, label=f"{site}" + (" (faulty)" if bad else ""))
    ax.axhline(cfg["reliability"]["quarantine_below"], color=AXIS, lw=0.8)
    ax.set(ylim=(-0.03, 1.08), xlabel="round", title="Client reliability (trust)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    _style(ax)
    ax.legend(fontsize=7, ncol=2)
    render_figure(f, colors)
with b2:
    f, ax = fig(5, 3.2)
    piv = cl.pivot(index="site", columns="round", values="status").replace(
        {"active": 2, "skipped": 1, "dropped": 0}).infer_objects()
    cmap = matplotlib.colors.ListedColormap([ORANGE, GRID, BLUE])
    ax.imshow(piv.to_numpy(float), cmap=cmap, vmin=0, vmax=2, aspect="auto")
    dr = cl.pivot(index="site", columns="round", values="drift").to_numpy(bool)
    ys, xs = np.where(dr)
    ax.scatter(xs, ys, s=18, facecolor=SURFACE, edgecolor=INK, lw=1)
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_xticks(range(0, len(piv.columns), 4), piv.columns[::4])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set(xlabel="round", title="Participation: trained / skipped / dropped, o = drift")
    render_figure(f, colors)
with b3:
    f, axes = plt.subplots(1, 2, figsize=(5, 3.2), sharey=True)
    for ax, d, ttl in [(axes[0], cl_fa, "FedAvg"), (axes[1], cl_rel, "Reliability-aware")]:
        for j, (site, g) in enumerate(d.groupby("site")):
            bad = site in faulty
            w = g["weight"].where(g.status == "active")
            ax.plot(g["round"], w, color=FAULT if bad else HEALTHY_GREYS[j % 4], lw=1.2,
                    marker="o", ms=3, label=site)
        ax.set(xlabel="round", title=ttl)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _style(ax)
    axes[0].set_ylabel("aggregation weight")
    axes[1].legend(fontsize=7)
    f.suptitle("Aggregation weights (without event-aware skipping)", fontsize=10, fontweight="bold", x=0.02, ha="left")
    f.tight_layout()
    render_figure(f, colors)

with st.expander("Site table (latest round)"):
    st.dataframe(styled_table(health[["capacity_mw", "trust", "quality", "quarantined", "state"]].round(3), colors))

st.subheader("Operator warnings")
safety = kpi.get("safety", {})
c1, c2, c3 = st.columns(3)
c1.metric("Rejected updates", safety.get("rejected_updates", 0))
c2.metric("Training fallback rounds", safety.get("fallback_rounds", 0))
c3.metric("Model rollbacks", safety.get("rollback_rounds", 0))
st.caption("Historical simulation replay. Each forecast issues 30 minutes before its target. "
           "These indicators describe the simulation, rather than a live SCADA connection.")
if "forecast_age_minutes" in today:
    st.write(f"Forecast age at issue time: {now.forecast_age_minutes:.0f} minutes. "
             f"Margin mode: {now.margin_source.replace('_', ' ')}.")
if now.planned_gap_mw > 1e-9:
    st.warning(f"Available backup cannot cover the planned requirement: {now.planned_gap_mw:.2f} MW gap.")
if "calibration_warning" in today and pd.notna(now.calibration_warning) and now.calibration_warning:
    st.warning(f"Conservative reserve floor active: {now.calibration_warning.replace('_', ' ')}.")
if "rolling_coverage" in today:
    st.write(f"Past margin coverage: {now.rolling_coverage:.1%} with "
             f"{int(now.calibration_samples)} usable samples. "
             f"Calibration age: {now.calibration_age_minutes:.0f} minutes.")
if "rejection_reason" in cl:
    with st.expander("Update decisions and reasons"):
        decisions = cl[(cl.rejected) | (cl.quarantined) | (cl.drift)]
        st.dataframe(styled_table(decisions[["round", "site", "trust", "quarantine_reason",
                                "rejection_reason", "drift"]], colors), hide_index=True)

with st.expander("Model, reliability and scale evidence"):
    tabs = st.tabs(["Model choice", "Reliability", "Scale", "Reserve outcomes"])
    def evidence(filename):
        path = Path("outputs/tables") / filename
        return pd.read_csv(path) if path.exists() else None
    with tabs[0]:
        frame = evidence("common_summary.csv")
        if frame is not None:
            st.caption("Five paired seeds. Lower daytime RMSE is better. Pooled methods require raw-data pooling.")
            evidence_chart(frame, "method:N", "rmse_mean:Q", colors, kind="bar")
            st.caption("Worst-site error appears separately in the table; it is not added to global RMSE.")
            st.dataframe(styled_table(frame.round(4), colors), hide_index=True)
        else:
            st.info("Run python run_common_comparison.py to generate model evidence.")
    with tabs[1]:
        frame = evidence("reliability_scorecard.csv")
        if frame is not None:
            st.dataframe(styled_table(frame.round(4), colors), hide_index=True)
        else:
            st.info("Run python run_reliability_stress.py to generate the scorecard.")
    with tabs[2]:
        frame = evidence("comm_scaling.csv")
        if frame is not None:
            st.caption("At 50–500 sites, repeated reference datasets test protocol traffic and runtime. "
                       "They do not establish geographic forecasting generalization. "
                       "Peak memory is cumulative for this process.")
            evidence_chart(frame, "n_sites:Q", "comm_mb_per_round:Q", colors, series="method:N")
            evidence_chart(frame, "n_sites:Q", "seconds:Q", colors, series="method:N")
            st.dataframe(styled_table(frame.round(4), colors), hide_index=True)
    with tabs[3]:
        frame = evidence("common_reserve_detail.csv")
        if frame is not None:
            cols = ["reserve_energy_mwh", "shortfall_energy_mwh", "availability_pct", "total_cost"]
            st.dataframe(styled_table(frame.groupby(["forecast_method", "policy"])[cols].mean().round(2), colors))
            st.caption("Synthetic demand and assumed costs/asset limits. The daily LP is a retrospective "
                       "schedule benchmark. Field deployment needs rolling planning with forecasts issued at that time.")
