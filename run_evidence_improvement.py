"""New chronological, causal benchmark; never overwrites mentor/legacy evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import importlib.metadata
import numpy as np
import pandas as pd
import yaml
from run_mentor_validation import india_config
from run_simulation import run_method
from src.data.nasa_power import read_power
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate, predictions_frame
from src.evaluation.serialization import dumps
from src.reserve.simulator import load_node, synthetic_demand
from src.reserve.reserve_policy import nsigma_reserve, empirical_reserve
from src.reserve.online import schedule_online, evaluate_schedule


def policy_margin(node, p):
    if p['family'] not in ('fixed', 'nsigma', 'empirical'):
        raise ValueError('Unknown reserve policy family')
    fallback = node.demand_mw.to_numpy()*.20
    if p['family'] == 'fixed':
        if not np.isfinite(p['fraction']) or p['fraction'] < 0:
            raise ValueError('Fixed fraction must be finite and nonnegative')
        return node.demand_mw.to_numpy()*p['fraction']
    if not 0 < p['delta'] < 1 or not isinstance(p['window'], int) or p['window'] < 3:
        raise ValueError('Tail probability must be in (0, 1); window must be >= 3')
    if not np.isfinite(p.get('floor', 0)) or p.get('floor', 0) < 0:
        raise ValueError('Margin floor must be finite and nonnegative')
    fn = nsigma_reserve if p['family'] == 'nsigma' else empirical_reserve
    margin = fn(node, p['delta'], p['window'], 3, 1, fallback=fallback)[0]
    return np.maximum(margin, p.get('floor', 0)*node.demand_mw.to_numpy())


def select_policy(validation):
    """Require no worse availability AND ENS in every paired validation seed.

    Among admissible candidates minimize mean scheduled backup, then cost, then name.
    Fixed is included, so there is always a safe validation fallback. No test input.
    """
    base = validation[validation.policy == 'fixed_20pct'].set_index('seed')
    if base.empty or not base.index.is_unique:
        raise ValueError('A unique fixed-reserve validation row is required for each seed')
    eligible = []
    for name, rows in validation.groupby('policy'):
        if rows.seed.duplicated().any() or set(rows.seed) != set(base.index):
            raise ValueError('Unpaired validation seeds')
        rows = rows.set_index('seed').reindex(base.index)
        if rows[['availability_pct', 'ens_mwh']].isna().any().any():
            raise ValueError('Unpaired validation seeds')
        if ((rows.availability_pct >= base.availability_pct-1e-9) &
                (rows.ens_mwh <= base.ens_mwh+1e-9)).all():
            eligible.append((rows.backup_mwh.mean(), rows.cost_score.mean(), name))
    if not eligible:
        raise ValueError('Missing baseline validation fallback')
    return min(eligible)[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/evidence_improvement.yaml')
    ap.add_argument('--out', default='outputs/evidence_improvement')
    args = ap.parse_args()
    spec = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    # Refuse to overwrite an already evaluated experiment. Reproduction uses a new directory.
    if (out/'test_results.csv').exists():
        raise ValueError('Results already exist; use --out with a new directory')
    out.mkdir(parents=True, exist_ok=True)
    frozen_dir = Path('outputs/mentor_validation')
    frozen_before = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_dir.glob('*') if p.is_file()}
    weather, provenance = read_power(spec['source'])
    weather_path = out/'weather.csv'; weather.to_csv(weather_path, index=False)
    cfg = india_config(weather_path)
    cfg['split']['mode'] = 'chronological'
    cfg['federated']['rounds'] = spec['rounds']
    cfg['model']['epochs'] = spec['local_epochs']
    (out/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out/'experiment.yaml').write_text(yaml.safe_dump(spec, sort_keys=False))
    cache, forecast_rows, site_rows, validation, split_info = {}, [], [], [], {}
    identity = {}
    for seed in spec['seeds']:
        clients, split_info = prepare_clients(cfg, spec['clients'], seed)
        for c in clients:
            assert pd.to_datetime(c.train.time).max() < pd.to_datetime(c.val.time).min()
            assert pd.to_datetime(c.val.time).max() < pd.to_datetime(c.test.time).min()
        for method in spec['methods']:
            start = time.perf_counter()
            res = run_method(method, clients, cfg, seed, verbose=False)
            m = evaluate(res, clients)
            forecast_rows.append(dict(seed=seed, method=method, rmse=m['global_rmse'],
                mae=m['global_mae'], healthy_rmse=m['healthy_rmse'],
                comm_mb=m.get('total_comm_mb', 0), seconds=time.perf_counter()-start))
            site_rows.extend(dict(seed=seed, method=method, site=k, **v) for k,v in m['per_site'].items())
            node = load_node(predictions_frame(res, clients))
            node['demand_mw'] = synthetic_demand(node.time, node.capacity_mw.iloc[0], cfg['reserve']['demand'], seed)
            paired = node[['time', 'split', 'daytime', 'actual_mw', 'demand_mw', 'capacity_mw']]
            if seed in identity:
                pd.testing.assert_frame_equal(identity[seed], paired)
            else:
                identity[seed] = paired.copy()
            node.to_csv(out/f'node_{seed}_{method}.csv', index=False)
            cache[seed, method] = node
            # Crucially policy selection sees validation rows ONLY, not test actuals/forecasts.
            v = node[node.split == 'val'].reset_index(drop=True)
            for p in spec['policies']:
                schedule = schedule_online(v, policy_margin(v, p), cfg['reserve'])
                validation.append(dict(seed=seed, method=method, policy=p['name'],
                                       **evaluate_schedule(schedule, cfg['reserve'], 'val')))
            print(f'{seed} {method}: RMSE={100*m["global_rmse"]:.3f}% ({time.perf_counter()-start:.1f}s)', flush=True)
    val = pd.DataFrame(validation)
    val.to_csv(out/'validation_candidates.csv', index=False)
    selected = {method:select_policy(val[val.method == method]) for method in spec['methods']}
    freeze = dict(selected=selected, selection='Every validation seed: availability >= fixed and ENS <= fixed; then minimum backup',
                  candidates=spec['policies'], seeds=spec['seeds'], test_used_for_selection=False)
    freeze_bytes = dumps(freeze, indent=2).encode()
    # Freeze to disk before the first operational test evaluation.
    (out/'policy_freeze.json').write_bytes(freeze_bytes)
    freeze_hash = hashlib.sha256(freeze_bytes).hexdigest()
    rows = []
    for (seed, method), node in cache.items():
        names = {'fixed_20pct', 'nominal_05', selected[method]}
        for p in spec['policies']:
            if p['name'] not in names:
                continue
            d = schedule_online(node, policy_margin(node, p), cfg['reserve'])
            daily = (d.scheduled_backup_mw*d.step_hours).groupby(d.time.dt.date).sum()
            assert (daily <= d.backup_energy_limit_mwh.iloc[0]+1e-9).all()
            rows.append(dict(seed=seed, method=method, policy=p['name'], selected=p['name']==selected[method],
                             **evaluate_schedule(d, cfg['reserve'], 'test')))
            if seed == spec['seeds'][0] and method == 'reliability_fedavg' and p['name'] == selected[method]:
                d.to_csv(out/'selected_schedule.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'test_results.csv', index=False)
    pd.DataFrame(forecast_rows).to_csv(out/'forecast_results.csv', index=False)
    pd.DataFrame(site_rows).to_csv(out/'sitewise_results.csv', index=False)
    frozen_after = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen_dir.glob('*') if p.is_file()}
    assert frozen_before == frozen_after, 'Frozen reference evidence changed'
    manifest = dict(provenance=provenance, splits=split_info, policy_freeze_sha256=freeze_hash,
        experiment_config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        dependencies={name:importlib.metadata.version(name) for name in
                      ['numpy','pandas','scipy','scikit-learn','torch','PyYAML']},
        frozen_reference_unchanged=True, reference_hashes=frozen_before,
        planner='Causal one-interval analytic LP, scheduled-energy debit; not retrospective full-day LP',
        assumptions=['Synthetic demand known at decision time (ideal load forecast)',
                     'Fixed grid availability known at decision time', 'One co-located weather record; modeled PV',
                     'Online residuals use only earlier observed targets',
                     'Validation labels select models and policy; test never selects settings',
                     'Daytime targets only; nightly backup adequacy not evaluated'],
        budgets={'local_only':f'{spec["local_epochs"]} full local epochs per site, best validation checkpoint',
                 'federated':f'{spec["rounds"]} rounds, <=20 selected sites/round, 1 local epoch/round; not equal total compute',
                 'persistence':'No training'},
        formal_privacy=False, field_validated=False)
    (out/'manifest.json').write_text(dumps(manifest, indent=2))
    print(pd.DataFrame(rows).groupby(['method', 'policy'])[['availability_pct','ens_mwh','backup_mwh','cost_score']].mean().to_string(), flush=True)


if __name__ == '__main__':
    main()
