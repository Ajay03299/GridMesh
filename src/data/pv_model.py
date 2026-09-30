"""Simplified PV power model: turns REAL irradiance + temperature into MODELED PV power.

This is a standard textbook approximation (PVWatts-style), not measured generation:

    T_cell = T_air + (NOCT - 20) / 800 * GHI
    P_pu   = derate * (GHI / G_ref) * (1 + temp_coeff * (T_cell - 25))
    P_pu   = clip(P_pu, 0, inverter_clip)

P_pu is per-unit of installed capacity (kW per kWp). Simplifications (documented in README):
horizontal plane (no tilt/POA transposition), no spectral/angle-of-incidence losses.
"""
import numpy as np


def pv_power_pu(ghi, temp_air, derate, temp_coeff, cfg_pv):
    ghi = np.clip(np.asarray(ghi, dtype=float), 0, None)
    t_cell = np.asarray(temp_air, dtype=float) + (cfg_pv["noct"] - 20.0) / 800.0 * ghi
    p = derate * (ghi / cfg_pv["ref_irradiance"]) * (1.0 + temp_coeff * (t_cell - 25.0))
    return np.clip(p, 0.0, cfg_pv["inverter_clip"])
