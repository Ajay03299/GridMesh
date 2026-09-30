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
    pv = pv_power_pu(base_df["GHI"], base_df["Temperature"], p.derate, p.temp_coeff, cfg["pv_model"])
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


def apply_fault(site_df, fault_type, fault_cfg, seed, site_id):
    """Return a DEGRADED copy of one site's frame. Simulated sensor/meter problems only."""
    rng = np.random.default_rng([seed, site_id, 99])
    df = site_df.copy()
    df.attrs = dict(site_df.attrs)
    n, c = len(df), fault_cfg[fault_type]
    if fault_type == "feature_corruption":
        for col in IRRADIANCE_COLS:
            df[col] = np.clip(df[col] * c["irradiance_scale"]
                              * (1 + rng.normal(0, c["noise_std"], n)), 0, None)
    elif fault_type == "target_noise":
        df["pv"] = np.clip(df["pv"] + rng.normal(0, c["std"], n) * (df["pv"] > 0), 0, 1)
    elif fault_type == "stale":
        idx = (np.arange(n) // c["hold_steps"]) * c["hold_steps"]
        cols = IRRADIANCE_COLS + OTHER_SENSOR_COLS + ["pv"]
        df[cols] = df[cols].to_numpy()[idx]
    elif fault_type == "bias":
        df["pv"] = np.clip(df["pv"] * c["scale"], 0, 1)
    else:
        raise ValueError(f"fault_type must be one of {FAULT_TYPES}")
    return df
