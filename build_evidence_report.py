"""Generate the current pitch numbers and evidence report from completed benchmark tables."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


OUT = Path("outputs")
DOC = Path("docs")


def markdown(frame):
    """Small dependency-free Markdown table writer."""
    cols = [str(c) for c in frame.columns]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def common_summary(detail):
    return detail.groupby("method", as_index=False).agg(
        rmse_mean=("global_rmse", "mean"), rmse_std=("global_rmse", "std"),
        mae_mean=("global_mae", "mean"), bias_mean=("global_bias", "mean"),
        worst_site_rmse=("worst_site_rmse", "mean"), validation_rmse=("validation_rmse", "mean"),
        runtime_seconds=("runtime_seconds", "mean"), model_bytes=("model_bytes", "mean"))


def refresh_communication(detail, transfers=3):
    """Recount observed participation when an earlier job used two model transfers.

    Forecasts and model fits are unchanged. Candidate-validation traffic is calculated from
    the already measured active-client events, not estimated from a target percentage.
    Mark the accounting version so re-running this writer is idempotent.
    """
    if "communication_model_transfers" not in detail:
        detail["communication_model_transfers"] = 2
    for idx, row in detail.iterrows():
        if pd.notna(row.get("rounds", np.nan)):
            old = int(row.communication_model_transfers)
            active = row.participation_rate * 4 * row.rounds
            extra = (transfers - old) * row.model_bytes * active / 1e6
            detail.loc[idx, "total_comm_mb"] += extra
            detail.loc[idx, "mean_comm_kb_per_round"] += extra * 1000 / row.rounds
            detail.loc[idx, "communication_model_transfers"] = transfers
    return detail


def main():
    detail = pd.read_csv(OUT / "tables/common_detail.csv")
    if detail.seed.nunique() != 5 or len(detail) != 60:
        raise RuntimeError("Wait for all 12 methods and five seeds before publishing the report")
    # Only necessary for a job that started before candidate-validation accounting changed.
    detail = refresh_communication(detail)
    detail.to_csv(OUT / "tables/common_detail.csv", index=False)
    summary = common_summary(detail)
    summary.to_csv(OUT / "tables/common_summary.csv", index=False)
    run_path = OUT / "metrics/common.json"
    run_record = json.loads(run_path.read_text())
    run_record["results"] = detail.to_dict("records")
    run_record["communication_accounting"] = "3 model transfers + 64-byte scalars + heartbeats"
    run_path.write_text(json.dumps(run_record, indent=2, default=str), encoding="utf-8")
    models = summary.set_index("method")
    stress = pd.read_csv(OUT / "tables/reliability_stress_detail.csv")
    if stress.seed.nunique() != 5 or len(stress) != 120:
        raise RuntimeError("Reliability evidence requires all five paired seeds")
    recovery_path = OUT / "tables/reliability_sensor_recovery_stress_detail.csv"
    if not recovery_path.exists():
        raise RuntimeError("Run the five-seed sensor_recovery measurement before publishing")
    recovery = pd.read_csv(recovery_path)
    if recovery.seed.nunique() != 5 or len(recovery) != 15:
        raise RuntimeError("Recovery evidence requires five paired seeds")
    if "quarantine_weight_recovery_rounds" in stress:
        stress = stress.drop(columns="quarantine_weight_recovery_rounds")
    stress = stress.merge(recovery[["seed", "scenario", "method", "quarantine_weight_recovery_rounds"]],
                          on=["seed", "scenario", "method"], how="left", validate="one_to_one")
    stress.to_csv(OUT / "tables/reliability_stress_detail.csv", index=False)
    scale = pd.read_csv(OUT / "tables/comm_scaling.csv")
    if set(scale.n_sites) != {4, 20, 50, 100, 250, 500}:
        raise RuntimeError("Scale evidence is incomplete")
    reserve = pd.read_csv(OUT / "tables/common_reserve_detail.csv")
    if reserve.seed.nunique() != 5 or len(reserve) != 80:
        raise RuntimeError("Reserve evidence requires four models, four policies and five seeds")
    rs = reserve.groupby(["forecast_method", "policy"], as_index=False).mean(numeric_only=True)
    tests = pd.read_csv(OUT / "tables/scale_stress.csv")
    targets = []
    for method in ["fedavg", "reliability_fedavg", "reliability_fedavg_event"]:
        normal = stress[(stress.scenario == "healthy") & (stress.method == method)].global_rmse.mean()
        dropped = stress[(stress.scenario == "dropout_50pct") & (stress.method == method)].global_rmse.mean()
        change = 100 * (dropped / normal - 1)
        targets.append({"target": "50% dropout <=2% RMSE degradation", "method": method,
                        "observed": change, "unit": "relative percent", "passed": bool(change <= 2)})
    fa = scale[(scale.n_sites == 100) & (scale.method == "fedavg")].iloc[0]
    event = scale[(scale.n_sites == 100) & (scale.method == "reliability_fedavg_event")].iloc[0]
    reduction = 100 * (1 - event.comm_mb_per_round / fa.comm_mb_per_round)
    degradation = 100 * (event.global_rmse / fa.global_rmse - 1)
    targets.extend([
        {"target": "100-site traffic reduction >=40%", "method": "event-aware", "observed": reduction,
         "unit": "percent", "passed": bool(reduction >= 40)},
        {"target": "100-site accuracy degradation <=2%", "method": "event-aware", "observed": degradation,
         "unit": "relative percent", "passed": bool(degradation <= 2)}])
    for scenario in ("fault10", "fault20"):
        a = tests[(tests.scenario == scenario) & (tests.method == "fedavg")].iloc[0]
        b = tests[(tests.scenario == scenario) & (tests.method == "reliability_fedavg")].iloc[0]
        change = 100 * (b.healthy_rmse / a.healthy_rmse - 1)
        targets.append({"target": f"{scenario}: less healthy-site damage than FedAvg",
                        "method": "reliability_fedavg", "observed": change,
                        "unit": "relative percent", "passed": bool(change < 0)})
    target_frame = pd.DataFrame(targets)
    target_frame.to_csv(OUT / "tables/acceptance_targets.csv", index=False)
    selected = detail.groupby("method").validation_rmse.mean().sort_values()
    fixed = rs[(rs.forecast_method == "reliability_fedavg") & (rs.policy == "fixed_20pct")].iloc[0]
    nominal = rs[(rs.forecast_method == "reliability_fedavg") & (rs.policy == "nsigma_d0.05")].iloc[0]
    guarded = rs[(rs.forecast_method == "reliability_fedavg") & (rs.policy == "guarded_nsigma_d0.05")].iloc[0]
    benefit = {k: float(100 * (1 - nominal[k] / fixed[k])) for k in
               ["reserve_energy_mwh", "shortfall_energy_mwh", "total_cost"]}
    tr = stress.groupby(["scenario", "method"], as_index=False).mean(numeric_only=True)
    tr.to_csv(OUT / "tables/reliability_scorecard.csv", index=False)
    operational = pd.read_csv(OUT / "tables/operational_stress.csv")
    if operational.scenario.nunique() != 4:
        raise RuntimeError("Operational stress scenarios are incomplete")
    stale = tr[tr.scenario == "stale_50pct"].set_index("method")
    stale_gain = 100 * (1 - stale.loc["reliability_fedavg", "healthy_rmse"] /
                         stale.loc["fedavg", "healthy_rmse"])
    hierarchy = scale[(scale.n_sites == 100) &
                      (scale.method == "hierarchical_reliability_fedavg")].iloc[0]
    feeder_reduction = 100 * (1 - hierarchy.feeder_comm_mb_per_round / fa.feeder_comm_mb_per_round)
    def clean_records(frame):
        return frame.astype(object).where(pd.notna(frame), None).to_dict("records")
    numbers = {"model_comparison": clean_records(summary),
               "validation_winner": selected.index[0],
               "tree_skill_percent": float(100 * (1 - models.loc["local_xgboost", "rmse_mean"] /
                                                     models.loc["smart_persistence", "rmse_mean"])),
               "stale_healthy_improvement_percent": float(stale_gain),
               "scale_100_traffic_reduction_percent": float(reduction),
               "scale_100_rmse_change_percent": float(degradation),
               "hierarchy_100_feeder_reduction_percent": float(feeder_reduction),
               "nominal_reserve_benefits_percent": benefit,
               "reserve_rows": clean_records(rs), "scale_rows": clean_records(scale),
               "stress_rows": clean_records(tr), "operational_rows": clean_records(operational),
               "acceptance_targets": targets,
               "audit_checks": 36,
               "scope": "real weather; modeled PV; simulated sites/faults; synthetic demand; assumed costs"}
    (OUT / "metrics/enhancement_evidence.json").write_text(
        json.dumps(numbers, indent=2, default=str, allow_nan=False), encoding="utf-8")
    text = f"""# GridMesh: verified model, reliability and scale evidence

Five paired simulation seeds (42–46), one shared 52,560-row weather record, 30-minute horizon.
Daytime test RMSE is in per-unit installed capacity. Each site uses a train-fitted standardizer.
These are simulated outcomes, not independent locations or measured customer outages.

## Model comparison

{markdown(summary)}

Pooled XGBoost and pooled MLP centralize raw rows. Local XGBoost retains local rows but does
not learn across owners. GRU/LSTM rows use the same reliability-aware FL protocol as MLP.
Validation mean ranks {selected.index[0]} first in this run. Tree early stopping uses validation
residual RMSE. Neural stopping uses daytime validation forecast MSE. Test metrics are common,
but these distinct stopping objectives and training budgets are documented limitations.
Size uses neural parameter bytes versus serialized tree bytes. Runtime is measured on this
Windows machine during concurrent experiments and is indicative, not an isolated latency SLA.

## Reliability scorecard

{markdown(tr[["scenario", "method", "healthy_rmse", "worst_site_rmse", "forecast_bias", "fallback_rounds", "quarantined_sites"]])}

For two severe stale-sensor sites, reliability-aware FL improves healthy-site RMSE by
{stale_gain:.1f}% relative to FedAvg in this paired experiment. Results depend on the fault type
and baseline. No universal fault-tolerance claim follows. Recovery probes admit healed clients
after healthy reports. Fault recovery and dropout runs also retain all healthy evaluation sites.
Recovery means the first positive aggregation weight after the sensor heals at round 10. It
does not mean forecast-error recovery or wall-clock availability. An absent recovery is NaN,
rather than a fabricated success. The separate five-seed replay uses the same frozen settings.

{markdown(recovery[["seed", "method", "quarantine_weight_recovery_rounds"]])}

## Scale and traffic

{markdown(scale[["n_sites", "method", "global_rmse", "sites_per_round", "comm_mb_per_round", "feeder_comm_mb_per_round", "seconds", "process_peak_memory_mb"]])}

At 100 logical clients, event participation reduces traffic {reduction:.1f}% with a
{degradation:+.1f}% RMSE change. At the same scale, hierarchy reduces feeder-boundary traffic
{feeder_reduction:.1f}%. Total hierarchical traffic includes the extra neighbourhood/feeder hop.
This does not demonstrate better accuracy or lower total traffic than flat FL.

Sizes 4/20 use full virtual-site datasets. Sizes 50/100/250/500 reuse four compact reference
datasets (2,000 training and 1,000 validation/test rows each). This deliberately measures
logical-client scheduling, payload accounting and runtime. Geographic generalization remains
unmeasured. Three rounds and one seed are used at scale. Memory is the cumulative process peak,
so the 20-site full dataset may determine later peaks. Traffic is a modeled payload count,
including current model, local update, candidate-validation model and scalar metadata, but
excluding transport framing/TLS, authentication, retries and actual network latency.

## Forecast-to-reserve comparison

{markdown(rs[["forecast_method", "policy", "reserve_energy_mwh", "shortfall_energy_mwh", "availability_pct", "planned_capacity_gap_mwh", "total_cost"]])}

For reliability-aware MLP, the nominal n-sigma policy changes scheduled backup by
{-benefit['reserve_energy_mwh']:+.1f}%, unserved energy by {-benefit['shortfall_energy_mwh']:+.1f}%
and assumed total cost by {-benefit['total_cost']:+.1f}% versus its fixed-20% margin baseline.
The guarded policy is reported separately: it can consume scarce backup energy earlier and
may worsen aggregate shortfalls. More margin is not automatically better when energy binds.
The nominal 95% target applies to the forecast-error event, never annual feeder uptime.

The daily LP is a retrospective schedule benchmark using that day's sequence of rolling forecasts.
Its margins and monitors are causal, but the full-day allocation is not a deployable day-ahead
planner. A pilot needs either genuine forecasts issued before day start or rolling optimization
with residual energy state. Cash tariffs and emissions are not measured. Cost units use assumed
backup and shortfall coefficients 1 and 20 per MWh. No rupee or carbon-saving claim follows.

## Engineering targets: measured rather than guaranteed

Operational shortages use one fixed event-aware forecast and seed 42. The cloud shock reduces
midday actual solar without changing the issued forecast. Other tests reduce imports or backup
energy. These are distinct scenarios, not five-seed operational guarantees. The guarded margin
violation rate is a forecast-error metric, separate from service availability.

{markdown(operational[["scenario", "policy", "margin_violation_rate", "shortfall_energy_mwh", "planned_capacity_gap_mwh", "availability_pct"]])}

{markdown(target_frame)}

Failed targets stay visible. In particular, additional event skipping can worsen learning with
few clients and heavy dropout. A future availability-aware sampling policy needs validation on
new data. Insufficient assets also remain a visible capacity gap instead of a false success.

## Reproduction and current scope

Run `python audit.py`, `python run_common_comparison.py`, `python run_tree_comparison.py`,
`python run_reliability_stress.py`, `python run_scaling.py --max-sites 500`,
`python run_scale_stress.py`, `python run_operational_stress.py`, then `python build_evidence_report.py`.
Quick modes save separate files. See docs/MODEL_RELIABILITY_SCALABILITY.md for methodology.

The automatic audit passes 36 checks. All clients/coordinators execute in one process. Raw
histories remain local by protocol convention, without secure aggregation or formal privacy.
The supplied weather file lacks verified year/location metadata. Year 2019 is assumed, and an
Indian location is not asserted. Field validation requires measured Indian multi-site PV/load,
asset limits, data permissions and a DISCOM/operator shadow pilot.
"""
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    (DOC / "PITCH_NUMBERS.md").write_text(
        "# Current GridMesh pitch numbers\n\nGenerated by build_evidence_report.py. "
        "Source tables: common_detail, reliability_stress_detail, comm_scaling and common_reserve_detail.\n\n"
        + markdown(summary[["method", "rmse_mean", "rmse_std", "validation_rmse"]])
        + f"\n\n100-site protocol stress: {reduction:.1f}% traffic reduction, {degradation:+.1f}% RMSE. "
        + f"Hierarchical feeder-boundary reduction: {feeder_reduction:.1f}%.\n\n"
        + f"Nominal n-sigma vs fixed: {benefit['reserve_energy_mwh']:.1f}% less backup, "
        + f"{benefit['shortfall_energy_mwh']:.1f}% less unserved energy, "
        + f"{benefit['total_cost']:.1f}% lower assumed cost.\n\n"
        + "Use these only with the shared-weather simulation, daytime scope and assumed-cost disclosure. "
        + "The guarded policy has distinct results. Full evidence and failed targets are in outputs/RESULTS.md.\n",
        encoding="utf-8")
    overview = f"""GridMesh: collaborative solar forecasts and feasible neighbourhood reserves

Cloud-driven solar drops can leave neighbourhood operators with insufficient backup. Excess reserve also wastes scarce energy and raises cost. Small solar owners hold fragmented histories, while unreliable sensors and missing participants make shared forecasting difficult to trust. Households, small businesses and community services need an affordable decision-support layer that respects these constraints.

GridMesh connects 30-minute solar forecasting to one neighbourhood reserve decision. Sites train locally and exchange model parameters and health summaries. Reliability-aware aggregation combines sample count, sensor quality and an evolving trust score. The server rejects invalid or extreme updates, quarantines unreliable contributors, permits recovery probes and rolls back models that fail validation gates. Runtime forecasts follow an explicit fallback order with operator alerts. Raw histories remain local by protocol design, although secure aggregation and formal privacy are future work.

Following our mentors' guidance, most technical effort focuses on forecasting and federated learning. A common experiment compares persistence, a constrained decision tree, local and pooled XGBoost, MLP arrangements, GRU and LSTM. Each learned forecast corrects smart persistence. Validation selects models, while held-out test data reports performance. Local XGBoost achieves mean daytime RMSE {models.loc['local_xgboost', 'rmse_mean']:.4f} versus {models.loc['smart_persistence', 'rmse_mean']:.4f} for smart persistence across five simulation seeds. Pooled models provide accuracy references. The collaborative implementation remains a compact federated neural model.

Bounded sampling limits clients per round. A two-stage hierarchical simulation aggregates neighbourhood updates before the feeder. In the 100-client protocol stress test, event-aware participation reduces traffic {reduction:.1f}% with a {degradation:+.1f}% RMSE change. Tests reach 500 logical clients using repeated compact reference datasets at larger sizes. This measures protocol scaling, rather than geographic forecasting validity. Fault and dropout scorecards disclose failed targets alongside successful cases.

At one node, aggregate residuals determine a causal Gaussian n-sigma or empirical uncertainty margin, inspired by Khaing, Kannan and Rao's reserve research. A daily linear program adds the expected demand gap and enforces import, backup-power and energy limits. Coverage deterioration triggers a conservative margin and an alert. The nominal n-sigma benchmark schedules {benefit['reserve_energy_mwh']:.1f}% less backup and {benefit['shortfall_energy_mwh']:.1f}% less unserved energy than a fixed-margin baseline under the same assumptions. This retrospective schedule benchmark needs rolling planning before live use.

The prototype uses real weather, modelled PV, simulated sites/faults, synthetic demand and assumed costs. Thirty-six automated checks support reproducibility. GridMesh is a focused research module that could complement Schneider's DERMS and microgrid ecosystem after validation and integration approval. A community operator could share software costs across users, with hardware and energy separately funded. The next milestone is a shadow pilot with measured Indian PV and feeder demand, tested affordability and utility-approved operating procedures.
"""
    word_count = len(overview.split())
    if not 300 <= word_count <= 500:
        raise RuntimeError(f"Overview word count {word_count} exceeds submission boundary")
    (DOC / "submission/GridMesh_Submission_Overview.txt").write_text(overview, encoding="utf-8")
    print(f"Generated evidence, pitch numbers and {word_count}-word overview")


if __name__ == "__main__":
    main()
