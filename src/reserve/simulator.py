"""One-node reserve scheduler and simulator.

The node aggregates all attached renewable sites:
    forecast_mw(t) = sum_k capacity_k * forecast_pu_k(t)
    actual_mw(t)   = sum_k capacity_k * actual_pu_k(t)
Capacities are SIMULATED site parameters; actual power is MODELED from real irradiance.

Planning separates the expected gap from forecast uncertainty:
    grid_t      = min(grid_limit, max(demand - forecast, 0))
    expected_t  = max(demand - grid_t - forecast, 0)
    required_t  = expected_t + uncertainty_margin_t

A linear program chooses backup q_t and planned uncovered requirement u_t:
    min sum_t (c_backup q_t + c_uncovered u_t) * dt
    s.t. q_t + u_t >= required_t, 0 <= q_t <= Pmax,
         sum_t q_t * dt <= Emax for each day.

Real-time evaluation deploys the committed backup against the realised net deficit.
Metrics are computed on DAYTIME TEST intervals only.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linprog

STEP_HOURS = 10 / 60


def load_node(pred):
    """Aggregate long-format predictions (CSV path or DataFrame) into one node series."""
    df = pred.copy() if isinstance(pred, pd.DataFrame) else pd.read_csv(pred, parse_dates=["time"])
    df["time"] = pd.to_datetime(df["time"])
    df["forecast_mw"] = df["capacity_mw"] * df["forecast_pu"]
    df["actual_mw"] = df["capacity_mw"] * df["actual_pu"]
    node = (df.groupby("time")
              .agg(split=("split", "first"), daytime=("daytime", "max"),
                   n_sites=("site", "nunique"), capacity_mw=("capacity_mw", "sum"),
                   forecast_mw=("forecast_mw", "sum"), actual_mw=("actual_mw", "sum"))
              .reset_index().sort_values("time").reset_index(drop=True))
    # contiguous blocks (val days + test days of one month); a gap > 3 h starts a new block
    node["block"] = (node["time"].diff() > pd.Timedelta(hours=3)).cumsum()
    return node


def synthetic_demand(times, total_capacity_mw, dcfg, seed):
    """SYNTHETIC load: base + morning and evening peaks + small noise. Not measured data."""
    rng = np.random.default_rng([seed, 5])
    t = pd.to_datetime(times)
    h = (t.dt.hour + t.dt.minute / 60).to_numpy()
    shape = (1 + dcfg["morning_peak"] * np.exp(-0.5 * ((h - 9.0) / 1.5) ** 2)
             + dcfg["evening_peak"] * np.exp(-0.5 * ((h - 19.5) / 2.0) ** 2))
    base = dcfg["base_mw_per_mw_capacity"] * total_capacity_mw
    return base * shape * (1 + rng.normal(0, dcfg["noise_std"], len(h)))


def _asset_limits(node, rcfg):
    if "capacity_mw" in node and node["capacity_mw"].notna().any():
        installed = float(node["capacity_mw"].dropna().iloc[0])
    else:
        installed = float(max(node["forecast_mw"].max(), node["demand_mw"].max(), 1.0))
    grid = rcfg.get("grid_import_limit_mw")
    grid = (rcfg.get("grid_import_limit_fraction", 0.5) * installed
            if grid is None else float(grid))
    power = rcfg.get("backup_power_limit_mw")
    power = (rcfg.get("backup_power_fraction", 0.35) * installed
             if power is None else float(power))
    energy = rcfg.get("backup_energy_limit_mwh")
    energy = (rcfg.get("backup_energy_hours", 2.0) * power
              if energy is None else float(energy))
    return max(grid, 0.0), max(power, 0.0), max(energy, 0.0)


def schedule_backup(node, uncertainty_margin_mw, rcfg):
    """Return an auditable, capacity-constrained day-ahead backup schedule."""
    d = node.copy()
    margin = np.maximum(np.asarray(uncertainty_margin_mw, float), 0.0)
    grid_limit, power_limit, energy_limit = _asset_limits(d, rcfg)
    d["uncertainty_margin_mw"] = margin
    d["grid_import_mw"] = np.minimum(
        grid_limit, np.maximum(d["demand_mw"] - d["forecast_mw"], 0.0))
    d["expected_gap_mw"] = np.maximum(
        d["demand_mw"] - d["grid_import_mw"] - d["forecast_mw"], 0.0)
    d["required_backup_mw"] = d["expected_gap_mw"] + margin
    # This prototype evaluates renewable-intermittency windows only.  Night service planning is a
    # different resource-adequacy problem and would otherwise consume the daily budget first.
    d.loc[~d["daytime"], ["expected_gap_mw", "required_backup_mw"]] = 0.0
    d["scheduled_backup_mw"] = 0.0
    d["planned_gap_mw"] = 0.0

    c_backup = float(rcfg["cost_reserve_per_mwh"])
    c_gap = float(rcfg["cost_shortfall_per_mwh"])
    peak_weight = float(rcfg.get("peak_priority_weight", 0.0))
    dates = pd.to_datetime(d["time"]).dt.date
    for _, idx in d.groupby(dates).groups.items():
        idx = list(idx)
        req = d.loc[idx, "required_backup_mw"].to_numpy(float)
        n = len(idx)
        # Variables are [q_0..q_n-1, u_0..u_n-1].  -q-u <= -requirement.
        # Without time-varying value, a binding daily energy budget makes the LP degenerate:
        # many schedules have the same total gap.  The configurable priority term allocates scarce
        # backup first to high-requirement intervals while keeping the problem linear and auditable.
        req_scale = req / max(req.max(), 1e-9)
        gap_value = c_gap * (1.0 + peak_weight * req_scale)
        objective = np.r_[np.full(n, c_backup * STEP_HOURS), gap_value * STEP_HOURS]
        A = np.zeros((n + 1, 2 * n))
        A[:n, :n] = -np.eye(n)
        A[:n, n:] = -np.eye(n)
        A[n, :n] = STEP_HOURS
        b = np.r_[-req, energy_limit]
        bounds = [(0.0, power_limit)] * n + [(0.0, None)] * n
        result = linprog(objective, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(f"Backup scheduler failed for {dates.iloc[idx[0]]}: {result.message}")
        d.loc[idx, "scheduled_backup_mw"] = result.x[:n]
        d.loc[idx, "planned_gap_mw"] = result.x[n:]

    d["grid_import_limit_mw"] = grid_limit
    d["backup_power_limit_mw"] = power_limit
    d["backup_energy_limit_mwh"] = energy_limit
    return d


def simulate(node, reserve_mw, rcfg, policy_name, delta=None):
    d = node.copy()
    d = schedule_backup(d, reserve_mw, rcfg)
    # Backward-compatible alias: reserve_mw now means scheduled dispatchable backup.
    d["reserve_mw"] = d["scheduled_backup_mw"]
    d["deficit_mw"] = np.maximum(
        d["demand_mw"] - d["grid_import_mw"] - d["actual_mw"], 0.0)
    d["used_mw"] = np.minimum(d["scheduled_backup_mw"], d["deficit_mw"])
    d["shortfall_mw"] = d["deficit_mw"] - d["used_mw"]
    ev = d[(d["split"] == "test") & d["daytime"]]

    reserve_mwh = ev["scheduled_backup_mw"].sum() * STEP_HOURS
    ens_mwh = ev["shortfall_mw"].sum() * STEP_HOURS
    cost_res = rcfg["cost_reserve_per_mwh"] * reserve_mwh
    cost_ens = rcfg["cost_shortfall_per_mwh"] * ens_mwh
    summary = {
        "policy": policy_name,
        "delta": delta,
        "intervals": int(len(ev)),
        "reserve_energy_mwh": reserve_mwh,
        "reserve_used_mwh": ev["used_mw"].sum() * STEP_HOURS,
        "shortfall_energy_mwh": ens_mwh,
        "ens_pct_of_demand": 100 * ens_mwh / (ev["demand_mw"].sum() * STEP_HOURS),
        "mean_reserve_mw": ev["scheduled_backup_mw"].mean(),
        "reserve_pct_of_forecast": 100 * ev["scheduled_backup_mw"].sum() / max(ev["forecast_mw"].sum(), 1e-9),
        "expected_gap_energy_mwh": ev["expected_gap_mw"].sum() * STEP_HOURS,
        "uncertainty_margin_energy_mwh": ev["uncertainty_margin_mw"].sum() * STEP_HOURS,
        "planned_capacity_gap_mwh": ev["planned_gap_mw"].sum() * STEP_HOURS,
        "grid_import_energy_mwh": ev["grid_import_mw"].sum() * STEP_HOURS,
        "backup_power_limit_mw": float(ev["backup_power_limit_mw"].iloc[0]),
        "backup_energy_limit_mwh_per_day": float(ev["backup_energy_limit_mwh"].iloc[0]),
        "availability_pct": 100 * (ev["shortfall_mw"] <= 1e-9).mean(),
        "target_availability_pct": None if delta is None else 100 * (1 - delta),
        "cost_reserve": cost_res,
        "cost_shortfall": cost_ens,
        "total_cost": cost_res + cost_ens,
    }
    return d, summary


def run_policies(node, cfg, seed):
    """Fixed reserve + n-sigma at every delta in the sweep. Returns [(timeseries, summary)]."""
    from src.reserve.reserve_policy import empirical_reserve, fixed_reserve, nsigma_reserve
    rcfg, h = cfg["reserve"], cfg["features"]["horizon"]
    node = node.copy()
    node["demand_mw"] = synthetic_demand(node["time"], node["capacity_mw"].iloc[0],
                                         rcfg["demand"], seed)
    fallback = fixed_reserve(node["demand_mw"], rcfg["fixed_fraction"])
    runs = [simulate(node, fallback, rcfg,
                     f"fixed_{int(round(rcfg['fixed_fraction'] * 100))}pct")]
    for dlt in rcfg["delta_sweep"]:
        r, mu, sd = nsigma_reserve(node, dlt, rcfg["window"], rcfg["min_periods"], h,
                                   fallback=fallback)
        d, s = simulate(node, r, rcfg, f"nsigma_d{dlt}", delta=dlt)
        d["mu_e"], d["sigma_e"] = mu, sd
        runs.append((d, s))
        r, q = empirical_reserve(node, dlt, rcfg["empirical_window"],
                                 rcfg["min_periods"], h, fallback=fallback)
        d, s = simulate(node, r, rcfg, f"empirical_d{dlt}", delta=dlt)
        d["empirical_loss_quantile_mw"] = q
        runs.append((d, s))
    return runs
