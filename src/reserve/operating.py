"""Advisory safety and ex-post operating uncertainty; no real asset control."""
import numpy as np
import pandas as pd


def advisory_inputs(node, *, max_age_minutes=60):
    """Invalid/stale solar becomes an explicitly labelled conservative bound.

    Missing demand/grid/assets cannot be repaired by guessing; scheduler validation
    rejects them. The zero-solar bound is not a forecast or an uptime guarantee.
    """
    if not np.isfinite(max_age_minutes) or max_age_minutes < 0:
        raise ValueError('Maximum age must be finite and nonnegative')
    d = node.copy()
    raw = pd.to_numeric(d.forecast_mw, errors='coerce')
    age = pd.to_numeric(d.get('forecast_age_minutes', pd.Series(0., index=d.index)), errors='coerce')
    missing = ~np.isfinite(raw) | (raw < 0)
    stale = ~np.isfinite(age) | (age < 0) | (age > max_age_minutes)
    warning = d.get('forecast_warning', pd.Series(False, index=d.index)).fillna(True).astype(bool)
    unsafe = missing | stale | warning
    d['raw_forecast_mw'] = raw
    d['forecast_mw'] = raw.mask(unsafe, 0.)
    d['forecast_warning'] = unsafe
    d['forecast_age_minutes'] = age
    d['forecast_source'] = np.where(unsafe, 'zero_solar_safety_bound', 'current_forecast')
    d['input_reason'] = np.select([missing, stale, warning],
        ['Missing/invalid solar forecast: conservative zero-solar bound; review required.',
         'Stale/invalid age: conservative zero-solar bound; refresh before approval.',
         'Degraded-site warning: conservative zero-solar bound; check site data.'], default='Fresh forecast')
    return d


def joint_error_margin(node, quantile, window=24, min_periods=6):
    """One-step delayed realized net-gap error; current/future actuals excluded.

    Assumes previous interval outcomes are observable before the next decision.
    No guarantee under delayed telemetry, abrupt shifts or finite assets.
    """
    if not 0 < quantile < 1 or not isinstance(window, int) or not 1 <= min_periods <= window:
        raise ValueError('Invalid rolling error calibration settings')
    planned_grid = node['planned_grid_mw']
    residual = ((node.actual_demand_mw-node.demand_mw)
                +(planned_grid-node.actual_grid_mw)
                +(node.forecast_mw-node.actual_mw))
    observed = residual.where(node.daytime).shift(1)
    # Reset at gaps/split transitions: do not carry artificial non-contiguous history.
    groups = ((node.time.diff() > pd.Timedelta(hours=3)) | node.split.ne(node.split.shift())).cumsum()
    calibrated = observed.groupby(groups, group_keys=False).transform(
        lambda v: v.rolling(window, min_periods=min_periods).quantile(quantile))
    return calibrated.fillna(node.demand_mw*.20).clip(lower=0).to_numpy()


def operating_metrics(schedule, config, split):
    """Actual demand/grid/delivery enter only here, after recommendations."""
    d = schedule[(schedule.split == split) & schedule.daytime].copy()
    if d.empty:
        raise ValueError('No operating evaluation intervals')
    cols=['actual_mw','actual_demand_mw','actual_grid_mw','actual_backup_fraction']
    values=d[cols].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any() or (d.actual_backup_fraction > 1).any():
        raise ValueError('Invalid ex-post operating observations')
    actual_backup=d.scheduled_backup_mw*d.actual_backup_fraction
    deficit=(d.actual_demand_mw-d.actual_grid_mw-d.actual_mw).clip(lower=0)
    ens=(deficit-actual_backup).clip(lower=0)
    backup=float((d.scheduled_backup_mw*d.step_hours).sum())
    energy=float((ens*d.step_hours).sum())
    return dict(intervals=len(d),availability_pct=float(100*(ens <= 1e-9).mean()),
        ens_mwh=energy,backup_mwh=backup,
        used_backup_mwh=float((np.minimum(actual_backup,deficit)*d.step_hours).sum()),
        planned_uncovered_mwh=float((d.planned_gap_mw*d.step_hours).sum()),
        margin_undercoverage_intervals=int(((deficit>d.required_backup_mw+1e-9)&(ens>1e-9)).sum()),
        asset_limited_intervals=int(((d.planned_gap_mw>1e-9)&(ens>1e-9)).sum()),
        energy_exhausted_intervals=int(((d.remaining_energy_before_mwh<=1e-9)&(ens>1e-9)).sum()),
        delivery_failure_intervals=int(((d.actual_backup_fraction<1)&(ens>1e-9)).sum()),
        cost_score=config['cost_reserve_per_mwh']*backup+config['cost_shortfall_per_mwh']*energy)
