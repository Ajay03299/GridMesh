"""One-node reserve simulator.

The node aggregates all attached renewable sites:
    forecast_mw(t) = sum_k capacity_k * forecast_pu_k(t)
    actual_mw(t)   = sum_k capacity_k * actual_pu_k(t)
Capacities are SIMULATED site parameters; actual power is MODELED from real irradiance.

Each interval, with reserve r_t held:
    deficit   = max(forecast - actual, 0)       renewables below forecast
    used      = min(r_t, deficit)               reserve deployed
    shortfall = deficit - used                  -> energy not served (ENS)
Metrics are computed on DAYTIME TEST intervals only.
"""
import numpy as np
import pandas as pd

STEP_HOURS = 10 / 60


def load_node(pred_csv):
    """Aggregate a long-format prediction file (from run_simulation.py) into one node series."""
    df = pd.read_csv(pred_csv, parse_dates=["time"])
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


def simulate(node, reserve_mw, rcfg, policy_name, delta=None):
    d = node.copy()
    d["reserve_mw"] = reserve_mw
    d["deficit_mw"] = np.maximum(d["forecast_mw"] - d["actual_mw"], 0.0)
    d["used_mw"] = np.minimum(d["reserve_mw"], d["deficit_mw"])
    d["shortfall_mw"] = d["deficit_mw"] - d["used_mw"]
    ev = d[(d["split"] == "test") & d["daytime"]]

    reserve_mwh = ev["reserve_mw"].sum() * STEP_HOURS
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
        "mean_reserve_mw": ev["reserve_mw"].mean(),
        "reserve_pct_of_forecast": 100 * ev["reserve_mw"].sum() / max(ev["forecast_mw"].sum(), 1e-9),
        "availability_pct": 100 * (ev["shortfall_mw"] <= 1e-9).mean(),
        "target_availability_pct": None if delta is None else 100 * (1 - delta),
        "cost_reserve": cost_res,
        "cost_shortfall": cost_ens,
        "total_cost": cost_res + cost_ens,
    }
    return d, summary
