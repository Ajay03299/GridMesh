"""Verify frozen reference schedules; summarize new evidence without changing experiments."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import numpy as np
import pandas as pd
from src.reserve.online import evaluate_schedule
import yaml


METRICS = ['availability_pct', 'ens_mwh', 'backup_mwh', 'cost_score']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/evidence_improvement')
    args = ap.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    ref = Path('outputs/mentor_validation')
    cfg = yaml.safe_load((ref/'config.yaml').read_text())
    d = pd.read_csv(ref/'reserve_schedule_comparison.csv', parse_dates=['time'])
    stored = pd.read_csv(ref/'model_to_operations_comparison.csv')
    rows = []
    for (seed, model, policy), frame in d.groupby(['seed','model','policy']):
        s = evaluate_schedule(frame, cfg['reserve'], 'test')
        old = stored[(stored.seed==seed)&(stored.model==model)&(stored.policy==policy)].iloc[0]
        for new, key in [('availability_pct','availability_pct'), ('ens_mwh','shortfall_energy_mwh'),
                         ('backup_mwh','reserve_energy_mwh'), ('cost_score','total_cost')]:
            np.testing.assert_allclose(s[new], old[key], rtol=1e-8, atol=1e-9)
        ev = frame[(frame.split=='test') & frame.daytime]
        deficit = np.maximum(ev.demand_mw-ev.grid_import_mw-ev.actual_mw, 0)
        missed = deficit > ev.scheduled_backup_mw+1e-9
        under_required = deficit > ev.required_backup_mw+1e-9
        rows.append(dict(seed=seed, model=model, policy=policy, **s,
            missed_intervals=int(missed.sum()),
            missed_requirement_too_small=int((missed & under_required).sum()),
            missed_despite_adequate_requirement=int((missed & ~under_required).sum())))
    reference = pd.DataFrame(rows)
    # Explicitly verify fair fixed/nominal physical inputs for every model and seed.
    cols = ['time','split','daytime','actual_mw','demand_mw','grid_import_mw',
            'backup_power_limit_mw','backup_energy_limit_mwh']
    for _, group in d.groupby(['seed','model']):
        a = group[group.policy=='fixed_20pct'][cols].reset_index(drop=True)
        b = group[group.policy=='nsigma_d0.05'][cols].reset_index(drop=True)
        pd.testing.assert_frame_equal(a,b)
    reference.to_csv(root/'reference_verified.csv', index=False)
    mlp = reference[reference.model=='mlp']
    lines = ['# Evidence improvement results', '',
        '## Frozen reference verified (unchanged)', '',
        'Bengaluru 100 virtual sites, monthly blocked offline evaluation; retrospective full-day LP. '
        'Values below are per-seed daytime test means ± sample standard deviation. '
        'Availability is the percentage of fully supplied intervals, not measured uptime. Cost score uses assumed units, not INR.', '']
    lines.extend(summary(mlp, 'policy'))
    lines += ['', '### Why nominal reserve loses reliability', '',
        'A tail-error margin is not a service guarantee. Small rolling windows and a Gaussian approximation '
        'can under-cover solar errors. Even when the requirement covers the actual deficit, power/daily-energy '
        'caps can prevent scheduling it. With constant uncovered-reserve penalties, the full-day LP can have '
        'multiple equally optimal time allocations: its objective does not directly maximize fully supplied '
        'intervals or minimize realized ENS. Neither actual solar nor a service-availability objective enters '
        'that planning LP. Wider margins alone therefore do not guarantee better service.', '']
    for policy, group in mlp.groupby('policy'):
        lines.append(f'- {policy}: mean {group.missed_intervals.mean():.1f} missed intervals/seed; '
                     f'{group.missed_requirement_too_small.mean():.1f} with too-small requirement; '
                     f'{group.missed_despite_adequate_requirement.mean():.1f} despite adequate requirement (allocation/asset limits).')
    lines += ['', '## Separate chronological experiment', '',
        'Train first 70%, validation next 15%, test final 15% of the hourly record; windows cannot cross split '
        'boundaries. Training precedes test. Single-interval analytic LP uses current forecast, known synthetic '
        'demand/grid, earlier observed residuals and remaining daily scheduled-energy budget; no future-day '
        'forecast sequence. Demand is assumed perfectly forecast, not a measured load model. '
        'This changes planning/split protocol and must NOT be compared as a direct improvement to the frozen reference.', '']
    path = root/'test_results.csv'
    if path.exists():
        test = pd.read_csv(path)
        lines += ['### Primary policy comparison: reliability_fedavg MLP, same causal planner', '']
        lines += summary(test[test.method=='reliability_fedavg'], 'policy')
        lines += ['', '### Selected policy for each forecasting baseline', '']
        lines += summary(test[test.selected], 'method')
        lines += ['', '### Paired policy test outcome (selected vs fixed, same chronological planner)', '']
        for method, g in test.groupby('method'):
            a = g[g.selected].set_index('seed').sort_index()
            b = g[g.policy=='fixed_20pct'].set_index('seed').sort_index()
            diff = a[METRICS]-b[METRICS]
            safe = ((diff.availability_pct>=-1e-9)&(diff.ens_mwh<=1e-9)).all()
            reduced = diff.backup_mwh.mean() < -1e-9
            lines.append(f'- {method}: selected {a.policy.iloc[0]}; availability delta '
                f'{diff.availability_pct.mean():+.3f} percentage points; ENS delta '
                f'{diff.ens_mwh.mean():+.5f} MWh; backup delta {diff.backup_mwh.mean():+.3f} MWh; '
                f'cost-score delta {diff.cost_score.mean():+.3f}. '
                f'No reliability degradation in every test seed: {bool(safe)}. Lower mean backup: {bool(reduced)}.')
        forecast = pd.read_csv(root/'forecast_results.csv')
        lines += ['', '### Forecast baselines', '',
            'Daytime test RMSE/MAE as % of installed site capacity. '
            'Local-only: 12 full epochs/site; FL: 12 rounds × up to 20 clients × 1 local epoch. '
            'These are documented budgets, not equal total compute. Basic FedAvg retains existing shared safety gates; '
            'it is a weighting baseline, not an entirely unguarded algorithm.', '']
        for method, g in forecast.groupby('method'):
            lines.append(f'- {method}: RMSE {100*g.rmse.mean():.3f} ± {100*g.rmse.std():.3f}%; '
                         f'MAE {100*g.mae.mean():.3f} ± {100*g.mae.std():.3f}%; '
                         f'mean modeled protocol traffic {g.comm_mb.mean():.3f} MB/run.')
        means = forecast.groupby('method')[['rmse','comm_mb']].mean()
        if {'fedavg','reliability_fedavg_event'}.issubset(means.index):
            payload = 100*(1-means.loc['reliability_fedavg_event','comm_mb']/means.loc['fedavg','comm_mb'])
            error = 100*(means.loc['reliability_fedavg_event','rmse']/means.loc['fedavg','rmse']-1)
            lines += ['', f'Selective FL vs basic FedAvg: {payload:.2f}% lower modeled protocol bytes/run '
                f'with {error:+.2f}% relative RMSE change (ratios of five-seed means; not physical-network measurements).',
                'The healthy-data reliability-weighted method is not more accurate than basic FedAvg on average. '
                'FL improves mean error versus smart persistence/local-only here, but smart persistence beats FL '
                'in seed 43. The selected persistence policy has lower mean ENS than selected FL despite lower '
                'fully-supplied availability and higher backup. No model is a universal operational winner.']
        lines += ['', '### Same-rule model comparison (nominal_05 for every method)', '']
        lines += summary(test[test.policy=='nominal_05'], 'method')
        lines += ['', 'Healthy co-located data alone does not prove fault-handling benefit. '
            'Use the separately labeled legacy stress experiments for their original robustness claims. '
            'Do not generalize selective-update savings to real networks or geographic generalization.']
        smoke_path = root/'failure_smoke.csv'
        if smoke_path.exists():
            smoke = pd.read_csv(smoke_path)
            lines += ['', '### Bounded current-India failure smoke', '',
                'Seed 42 only; 100 virtual sites; same chronological data. '
                'Twenty stale training copies or 50% random dropout; clean test targets. '
                'Policies remain frozen from the clean validation run; no fault/test re-tuning.', '']
            for scenario, group in smoke.groupby('scenario'):
                for _, r in group.iterrows():
                    lines.append(f'- {scenario}, {r["method"]}: healthy RMSE {100*r.healthy_rmse:.3f}%; '
                        f'fully supplied {r.availability_pct:.3f}%; ENS {r.ens_mwh:.4f} MWh; '
                        f'backup {r.backup_mwh:.3f} MWh; cost {r.cost_score:.3f}; traffic {r.comm_mb:.3f} MB.')
            lines += ['', 'This one-seed smoke cannot establish statistically general robustness or service uptime.']
        manifest = json.loads((root/'manifest.json').read_text())
        freeze_hash = hashlib.sha256((root/'policy_freeze.json').read_bytes()).hexdigest()
        assert freeze_hash == manifest['policy_freeze_sha256']
        current_reference = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ref.glob('*') if p.is_file()}
        assert current_reference == manifest['reference_hashes']
        verification = dict(policy_freeze_hash_verified=True, frozen_reference_unchanged=True,
            environment_versions={name:importlib.metadata.version(name) for name in
                                  ['numpy','pandas','scipy','scikit-learn','torch','PyYAML']},
            current_code_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path('run_evidence_improvement.py'),Path('src/reserve/online.py'),Path('configs/evidence_improvement.yaml')]})
        (root/'verification.json').write_text(json.dumps(verification, indent=2))
    else:
        lines += ['New experiment not yet completed; do not claim new results.']
    lines += ['', '## Evidence boundaries', '',
        'Virtual sites, one gridded-weather record, modeled PV, synthetic demand, assumed available generic '
        'backup, no measured outage hours, no formal privacy, no geographic holdout. Policy selected solely '
        'on validation and frozen before operational test evaluation. Test metrics are descriptive; five seeds '
        'are not an independent-field confidence interval. All disappointing outcomes remain in the report.']
    (root/'FINDINGS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


def summary(frame, key):
    lines = ['| Comparison | Fully supplied % | Realized ENS MWh | Scheduled backup MWh | Cost score |',
             '|---|---:|---:|---:|---:|']
    for name,g in frame.groupby(key):
        vals = [f'{g[c].mean():.4f} ± {g[c].std():.4f}' for c in METRICS]
        lines.append('| '+str(name)+' | '+' | '.join(vals)+' |')
    return lines


if __name__ == '__main__': main()
