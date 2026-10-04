"""Paired, auditable Indian-weather experiment. Does not overwrite legacy evidence.

python run_mentor_validation.py --source data/raw/nasa_power_bengaluru_2024.json
Full protocol: 100 independently perturbed co-located virtual sites, five seeds,
three neural backbones, same FL rules and reserve assets. Targets are MODELED PV.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.nasa_power import read_power
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate, predictions_frame
from src.evaluation.serialization import dumps
from src.reserve.simulator import load_node, run_policies


def india_config(path):
    cfg = load_config()
    cfg['data'].update(path=str(path), target_col=None, timestamp_col='timestamp')
    cfg['features'].update(lookback=6, horizon=1, lagged=['pv', 'GHI', 'kt', 'Temperature'],
                           current=[], cloud_type_values=[])
    cfg['model'].update(sequence_features=4, hidden=[32, 32])
    cfg['virtual_sites'].update(capacity_mw=[.002, .008], capacity_decimals=6)
    cfg['federated'].update(rounds=12, client_fraction=1., max_clients_per_round=20)
    cfg['reserve'].update(step_hours=1., grid_dispatch_mode='fixed_availability',
        peak_priority_weight=0., delta_sweep=[.05], window=6, empirical_window=12,
        min_periods=3)
    cfg['safety']['calibration_window'] = 12
    return cfg


def check_schedule(d):
    eps = 1e-8
    assert (d.scheduled_backup_mw >= -eps).all()
    assert (d.scheduled_backup_mw <= d.backup_power_limit_mw + eps).all()
    assert (d.scheduled_backup_mw + d.planned_gap_mw >= d.required_backup_mw - eps).all()
    daily = (d.scheduled_backup_mw*d.step_hours).groupby(d.time.dt.date).sum()
    assert (daily <= d.backup_energy_limit_mwh.iloc[0] + eps).all()
    assert (d.planned_gap_mw >= -eps).all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/raw/nasa_power_bengaluru_2024.json')
    ap.add_argument('--seeds', default='42,43,44,45,46')
    ap.add_argument('--clients', type=int, default=100)
    ap.add_argument('--rounds', type=int, default=12)
    args = ap.parse_args()
    out = Path('outputs/mentor_validation'); out.mkdir(parents=True, exist_ok=True)
    frame, provenance = read_power(args.source)
    data_path = out/'bengaluru_hourly.csv'; frame.to_csv(data_path, index=False)
    cfg = india_config(data_path); cfg['federated']['rounds'] = args.rounds
    (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out/'provenance.json').write_text(dumps(provenance, indent=2))
    seeds = [int(s) for s in args.seeds.split(',')]
    forecast_rows, site_rows, operations, schedules, site_params = [], [], [], [], []
    for seed in seeds:
        clients, _ = prepare_clients(cfg, args.clients, seed)
        site_params.extend({'seed':seed, **vars(c.params), 'train_rows':len(c.train.y),
                            'test_rows':len(c.test.y)} for c in clients)
        shared = None
        for architecture in ('mlp', 'gru', 'lstm'):
            mc = copy.deepcopy(cfg); mc['model']['architecture'] = architecture
            started = time.perf_counter()
            res = run_method('reliability_fedavg', clients, mc, seed, verbose=False)
            metric = evaluate(res, clients)
            forecast_rows.append({'seed':seed, 'model':architecture,
                'global_rmse':metric['global_rmse'], 'global_mae':metric['global_mae'],
                'worst_site_rmse':metric['worst_site_rmse'], 'total_comm_mb':metric['total_comm_mb'],
                'runtime_seconds':time.perf_counter()-started})
            site_rows.extend({'seed':seed, 'model':architecture, 'site':name, **m}
                             for name,m in metric['per_site'].items())
            node = load_node(predictions_frame(res, clients))
            for d,s in run_policies(node, mc, seed):
                if s['policy'] not in ('fixed_20pct', 'nsigma_d0.05'):
                    continue
                check_schedule(d)
                # Explicit assertion: downstream inputs other than forecast/margin are paired.
                identity = d[['time','actual_mw','capacity_mw','demand_mw','grid_import_mw',
                    'backup_power_limit_mw','backup_energy_limit_mwh']].reset_index(drop=True)
                if shared is None:
                    shared = identity
                else:
                    pd.testing.assert_frame_equal(shared, identity)
                operations.append({'seed':seed, 'model':architecture, **s})
                d['seed'],d['model'],d['policy'] = seed,architecture,s['policy']
                schedules.append(d)
            print(f"seed={seed} model={architecture} RMSE={100*metric['global_rmse']:.3f}% "
                  f"seconds={time.perf_counter()-started:.1f}", flush=True)
            pd.DataFrame(forecast_rows).to_csv(out/'forecasting_metrics.csv', index=False)
            pd.DataFrame(site_rows).to_csv(out/'sitewise_metrics.csv', index=False)
            pd.DataFrame(operations).to_csv(out/'model_to_operations_comparison.csv', index=False)
    schedule = pd.concat(schedules, ignore_index=True)
    schedule.to_csv(out/'reserve_schedule_comparison.csv', index=False)
    pd.DataFrame(site_params).to_csv(out/'virtual_site_parameters.csv', index=False)
    fm = pd.DataFrame(forecast_rows).groupby('model').agg(
        rmse=('global_rmse','mean'), rmse_sd=('global_rmse','std'), mae=('global_mae','mean'))
    op = pd.DataFrame(operations).groupby(['model','policy']).agg(
        fully_supplied_pct=('availability_pct','mean'), ens_mwh=('shortfall_energy_mwh','mean'),
        backup_mwh=('reserve_energy_mwh','mean'), cost_score=('total_cost','mean'),
        planned_uncovered_mwh=('planned_capacity_gap_mwh','mean'))
    # Choose chart model on validation/training protocol, not to optimize displayed outcome.
    # MLP is the existing proposed backbone and is fixed before this experiment.
    representative = schedule[(schedule.seed == seeds[0]) & (schedule.model == 'mlp') &
        (schedule.policy == 'nsigma_d0.05') & (schedule.split == 'test')]
    first_day = representative.time.dt.date.iloc[0]
    representative = representative[representative.time.dt.date == first_day]
    representative.to_csv(out/'representative_schedule.csv', index=False)
    summary = {'protocol':'India hourly v1', 'seeds':seeds,'n_sites':args.clients,
        'horizon_minutes':60,'rounds':args.rounds,'max_clients_per_round':20,
        'models':fm.reset_index().to_dict('records'),
        'operations':op.reset_index().to_dict('records'),
        'limitations':['one NASA POWER weather record, modeled PV, synthetic demand, assumed assets',
            '100 heterogeneous sites are emulated in one process; no secure aggregation',
            'retrospective per-day LP replay, not an online day-ahead dispatch trial',
            'daytime test intervals only, not annual uptime or measured outage hours',
            'hourly gridded weather, not independent measured Indian PV sites'],
        'metrics':'daytime test error in p.u. nameplate; RMSE and MAE as percent = 100*p.u.',
        'planning_vs_realized':'u is uncovered planned requirement; realised ENS uses actual solar',
        'cost_units':'assumed units, backup coefficient 1 and unserved coefficient 20; not rupees'}
    (out/'summary_metrics.json').write_text(dumps(summary, indent=2))
    (out/'optimization_summary.json').write_text(dumps({'solver':'scipy linprog HiGHS',
        'objective':'sum((1*q + 20*u)*dt)', 'step_hours':1,'peak_priority_weight':0,
        'grid_dispatch_mode':'fixed_availability','all_pairing_assertions_passed':True,
        'all_power_daily_energy_requirement_constraints_passed':True,
        'representative_date':str(first_day),'representative_daily_scheduled_mwh':
            float((representative.scheduled_backup_mw*representative.step_hours).sum()),
        'representative_daily_budget_mwh':float(representative.backup_energy_limit_mwh.iloc[0])},indent=2))
    print(dumps(summary,indent=2), flush=True)


if __name__ == '__main__':
    main()
