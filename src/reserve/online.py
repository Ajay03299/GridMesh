"""Causal one-interval allocation; separate from the frozen retrospective LP.

For constant c_uncovered > c_backup, the one-interval LP has the analytic
solution q=min(requirement, power cap, remaining daily energy/dt). No future
targets or forecasts enter this decision. This is myopic, not a optimal day plan.
Scheduled energy consumes the budget even if subsequently not used (conservative).
"""
import numpy as np
import pandas as pd
from src.reserve.simulator import _asset_limits


def schedule_online(node, margin, rcfg):
    d = node.copy().reset_index(drop=True)
    required = ['time', 'forecast_mw', 'demand_mw', 'daytime', 'split', 'capacity_mw']
    if any(c not in d for c in required):
        raise ValueError('Missing decision inputs')
    if d.empty or not pd.api.types.is_bool_dtype(d.daytime):
        raise ValueError('Need at least one decision and a boolean daytime mask')
    d['time'] = pd.to_datetime(d.time)
    if d.time.isna().any() or d.time.duplicated().any() or not d.time.is_monotonic_increasing:
        raise ValueError('Decision targets must be unique and chronological')
    capacity = d.capacity_mw.to_numpy(float)
    if not np.isfinite(capacity).all() or (capacity <= 0).any() or not np.allclose(capacity, capacity[0]):
        raise ValueError('Installed capacity must be a known, finite, constant positive asset input')
    m = np.asarray(margin, float)
    dt = float(rcfg.get('step_hours', 1))
    numeric = d[['forecast_mw', 'demand_mw']].to_numpy(float)
    if (m.shape != (len(d),) or not np.isfinite(m).all() or (m < 0).any()
            or not np.isfinite(numeric).all() or (numeric < 0).any()):
        raise ValueError('Decision inputs must be finite and nonnegative')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid interval duration')
    if rcfg.get('grid_dispatch_mode') != 'fixed_availability' or rcfg.get('peak_priority_weight', 0) != 0:
        raise ValueError('Online experiment requires fixed grid and constant objective coefficients')
    raw_limits = [rcfg.get(k) for k in ('grid_import_limit_mw', 'backup_power_limit_mw', 'backup_energy_limit_mwh')]
    if any(v is not None and (not np.isfinite(v) or v < 0) for v in raw_limits):
        raise ValueError('Asset limits must be finite and nonnegative')
    grid, power, energy = _asset_limits(d, rcfg)
    cb, cu = float(rcfg['cost_reserve_per_mwh']), float(rcfg['cost_shortfall_per_mwh'])
    if not np.isfinite([grid, power, energy, cb, cu]).all() or min(cb, cu) < 0:
        raise ValueError('Invalid assets or costs')
    d['grid_import_mw'] = np.minimum(grid, d.demand_mw)
    d['expected_gap_mw'] = np.maximum(d.demand_mw-d.grid_import_mw-d.forecast_mw, 0)
    d['uncertainty_margin_mw'] = m
    d['required_backup_mw'] = d.expected_gap_mw + m
    d.loc[~d.daytime, ['expected_gap_mw', 'required_backup_mw']] = 0
    remaining, current_date = energy, None
    qs, before = [], []
    for row in d.itertuples():
        if row.time.date() != current_date:
            current_date, remaining = row.time.date(), energy
        before.append(remaining)
        q = min(row.required_backup_mw, power, remaining/dt) if cu > cb else 0.0
        qs.append(q)
        remaining = max(remaining-q*dt, 0.0)
    d['scheduled_backup_mw'] = qs
    d['remaining_energy_before_mwh'] = before
    d['planned_gap_mw'] = np.maximum(d.required_backup_mw-d.scheduled_backup_mw, 0)
    d['backup_power_limit_mw'], d['backup_energy_limit_mwh'], d['step_hours'] = power, energy, dt
    d['backup_cost_coefficient'], d['uncovered_cost_coefficient'] = cb, cu
    return d


def evaluate_schedule(d, rcfg, split):
    """Ex-post evaluation only. Actual solar never enters schedule_online."""
    ev = d[(d.split == split) & d.daytime]
    if ev.empty:
        raise ValueError('No daytime evaluation intervals')
    actual = ev.actual_mw.to_numpy(float)
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError('Invalid actual solar for evaluation')
    ens = np.maximum(ev.demand_mw-ev.grid_import_mw-actual-ev.scheduled_backup_mw, 0)
    backup = float((ev.scheduled_backup_mw*ev.step_hours).sum())
    energy = float((ens*ev.step_hours).sum())
    return dict(intervals=len(ev), availability_pct=float(100*(ens <= 1e-9).mean()),
                ens_mwh=energy, backup_mwh=backup,
                planned_uncovered_mwh=float((ev.planned_gap_mw*ev.step_hours).sum()),
                cost_score=rcfg['cost_reserve_per_mwh']*backup+rcfg['cost_shortfall_per_mwh']*energy)


def advice(row):
    """Explain a recommendation without consulting actual solar or realized ENS."""
    req, q = float(row['required_backup_mw']), float(row['scheduled_backup_mw'])
    if req <= 1e-9:
        reason = 'No forecast gap or uncertainty reserve requirement in this interval.'
    elif row.get('uncovered_cost_coefficient', 20) <= row.get('backup_cost_coefficient', 1):
        reason = 'The configured uncovered-reserve penalty does not justify scheduling backup; review objective coefficients.'
    elif q + 1e-9 < req:
        reason = 'Requirement exceeds available backup power or remaining daily scheduled-energy budget.'
    else:
        reason = 'Requirement covered within the assumed asset limits.'
    age = float(row.get('forecast_age_minutes', 0))
    attention = bool(row.get('forecast_warning', False)) or not np.isfinite(age) or age < 0 or age > 60
    if row.get('forecast_source') == 'zero_solar_safety_bound':
        reason = str(row.get('input_reason', 'Conservative solar bound; review required.')) + ' ' + reason
    if q + 1e-9 < req:
        attention = True
    return dict(reason=reason, operator_attention=attention,
                action='Check asset availability and data; approve or reject advice. No equipment command is sent.',
                asset='Generic dispatchable backup with power and daily-energy limits; technology/access unconfirmed.')
