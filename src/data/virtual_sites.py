"""Virtual renewable sites built from ONE real weather record.

SIMULATED HETEROGENEITY (all ranges in config.yaml -> virtual_sites):
  * PV configuration  : installed capacity, derate, temperature coefficient
  * sensor quality    : Gaussian noise, irradiance calibration bias, random dropouts
  * power-meter noise : small Gaussian noise on the PV power target
  * sample volume     : each site owns only its most recent `history_fraction` of training history

The weather itself is real and shared (all sites sit at one aggregation node).
Site k's parameters depend only on (seed, k), so site 0..3 are identical whether you
run 4 or 100 clients.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.pv_model import pv_power_pu

IRRADIANCE_COLS = ["GHI", "DNI", "DHI"]                      # measured by pyranometers
OTHER_SENSOR_COLS = ["Temperature", "Relative Humidity", "Pressure", "Dew Point"]
# Clear-sky GHI, zenith angle, aerosol/ozone etc. are model/satellite products -> no sensor noise.


@dataclass
class SiteParams:
    site_id: int
    name: str
    capacity_mw: float
    derate: float
    temp_coeff: float
    sensor_noise_std: float
    sensor_bias: float
    missing_rate: float
    target_noise_std: float
    history_fraction: float


def _u(rng, lo_hi):
    lo, hi = lo_hi
    return float(rng.uniform(lo, hi))


def site_name(k):
    """0->A, 1->B, ... 25->Z, 26->S26 ..."""
    return chr(ord("A") + k) if k < 26 else f"S{k}"


def sample_site_params(k, vs_cfg, seed):
    rng = np.random.default_rng([seed, k])
    return SiteParams(
        site_id=k, name=site_name(k),
        capacity_mw=round(_u(rng, vs_cfg["capacity_mw"]), 2),
        derate=_u(rng, vs_cfg["derate"]),
        temp_coeff=_u(rng, vs_cfg["temp_coeff"]),
        sensor_noise_std=_u(rng, vs_cfg["sensor_noise_std"]),
        sensor_bias=_u(rng, vs_cfg["sensor_bias"]),
        missing_rate=_u(rng, vs_cfg["missing_rate"]),
        target_noise_std=_u(rng, vs_cfg["target_noise_std"]),
        history_fraction=_u(rng, vs_cfg["history_fraction"]),
    )


def build_site_frame(base_df, p: SiteParams, cfg, seed):
    """Return one site's DataFrame: noisy sensor features + modeled PV power `pv` (p.u.)."""
    rng = np.random.default_rng([seed, p.site_id, 1])
    df = base_df.copy()
    n = len(df)

    # 1) PV power from the TRUE (clean) weather — the plant sees the real sun.
    #    target.mode "column": use a MEASURED power column instead (scaled to p.u.).
    if cfg.get("target", {}).get("mode") == "column":
        pv = measured_pu(base_df, cfg)
    else:
        pv = pv_power_pu(base_df["GHI"], base_df["Temperature"], p.derate, p.temp_coeff,
                         cfg["pv_model"])
    meter_noise = rng.normal(0, p.target_noise_std, n) * (pv > 0)
    df["pv"] = np.clip(pv + meter_noise, 0, cfg["pv_model"]["inverter_clip"])

    # 2) The site's weather SENSORS are imperfect.
    for c in IRRADIANCE_COLS:
        noisy = df[c] * (1 + p.sensor_bias) * (1 + rng.normal(0, p.sensor_noise_std, n))
        df[c] = np.clip(noisy, 0, None)
    for c in OTHER_SENSOR_COLS:
        df[c] = df[c] + rng.normal(0, p.sensor_noise_std, n) * base_df[c].std()

    # 3) Random sensor dropouts -> NaN, then forward-fill (what a real SCADA pipeline does).
    sensor_cols = IRRADIANCE_COLS + OTHER_SENSOR_COLS
    mask = rng.random((n, len(sensor_cols))) < p.missing_rate
    vals = df[sensor_cols].to_numpy(dtype=float)
    vals[mask] = np.nan
    df[sensor_cols] = vals
    df.attrs["missing_frac"] = float(mask.mean())
    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    return df


def generate_virtual_sites(base_df, cfg, n_clients=4, seed=42):
    """Return list of (SiteParams, site DataFrame). Reproducible for a given seed."""
    out = []
    for k in range(n_clients):
        p = sample_site_params(k, cfg["virtual_sites"], seed)
        out.append((p, build_site_frame(base_df, p, cfg, seed)))
    return out


def params_table(sites):
    return pd.DataFrame([vars(p) for p, _ in sites]).set_index("name")


FAULT_TYPES = ("feature_corruption", "target_noise", "stale", "bias")


def apply_fault(site_df, fault_type, fault_cfg, seed, site_id, severity=1.0):
    """Return a DEGRADED copy of one site's frame. Simulated sensor/meter problems only.
    `severity` scales the fault: 0 = no fault, 1 = config values, 2 = twice as strong."""
    rng = np.random.default_rng([seed, site_id, 99])
    df = site_df.copy()
    df.attrs = dict(site_df.attrs)
    n, c, s = len(df), fault_cfg[fault_type], severity
    if fault_type == "feature_corruption":
        scale = max(0.0, 1 - (1 - c["irradiance_scale"]) * s)
        for col in IRRADIANCE_COLS:
            df[col] = np.clip(df[col] * scale * (1 + rng.normal(0, c["noise_std"] * s, n)), 0, None)
    elif fault_type == "target_noise":
        df["pv"] = np.clip(df["pv"] + rng.normal(0, c["std"] * s, n) * (df["pv"] > 0), 0, 1)
    elif fault_type == "stale":
        hold = max(1, int(round(c["hold_steps"] * s)))
        idx = (np.arange(n) // hold) * hold
        cols = IRRADIANCE_COLS + OTHER_SENSOR_COLS + ["pv"]
        df[cols] = df[cols].to_numpy()[idx]
    elif fault_type == "bias":
        df["pv"] = np.clip(df["pv"] * (1 + (c["scale"] - 1) * s), 0, 1)
    else:
        raise ValueError(f"fault_type must be one of {FAULT_TYPES}")
    return df


def measured_pu(df, cfg):
    """Measured power column -> p.u. of installed capacity (target.capacity, else column max)."""
    col = cfg["target"].get("column")
    if not col or col not in df.columns:
        raise ValueError(f"target.column '{col}' not found in the data")
    cap = cfg["target"].get("capacity") or float(df[col].max())
    return np.clip(df[col].to_numpy(float) / cap, 0, 1)


def real_sites(base_df, cfg):
    """REAL multi-site data (data.site_col is set): one client per site ID, no simulated noise.
    Every site needs the same weather columns as the main file, e.g. NSRDB weather downloaded
    for the site's coordinates and joined with the plant's measured power."""
    site_col, tcfg = cfg["data"]["site_col"], cfg["target"]
    sensor_cols = [c for c in IRRADIANCE_COLS + OTHER_SENSOR_COLS if c in base_df.columns]
    out = []
    groups = sorted(base_df.groupby(site_col), key=lambda g: str(g[0]))
    for k, (sid, g) in enumerate(groups):
        df = g.sort_values("timestamp").reset_index(drop=True)
        if tcfg.get("mode") == "column":
            if tcfg.get("capacity_col"):          # nameplate capacity per site (best)
                cap = float(df[tcfg["capacity_col"]].iloc[0])
            else:                                 # one global value, else the observed peak
                cap = tcfg.get("capacity") or float(df[tcfg["column"]].max())
            df["pv"] = np.clip(df[tcfg["column"]].to_numpy(float) / cap, 0, 1)
            cap_mw = cap * float(tcfg.get("column_unit_mw", 1.0))
        else:
            vs = cfg["virtual_sites"]
            df["pv"] = pv_power_pu(df["GHI"], df["Temperature"], float(np.mean(vs["derate"])),
                                   float(np.mean(vs["temp_coeff"])), cfg["pv_model"])
            cap_mw = 1.0
        df.attrs["missing_frac"] = float(df[sensor_cols].isna().to_numpy().mean())
        df[sensor_cols] = df[sensor_cols].ffill().bfill()
        nan = float("nan")
        out.append((SiteParams(site_id=k, name=str(sid), capacity_mw=round(cap_mw, 4),
                               derate=nan, temp_coeff=nan, sensor_noise_std=0.0,
                               sensor_bias=0.0, missing_rate=0.0, target_noise_std=0.0,
                               history_fraction=1.0), df))
    return out
