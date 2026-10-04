"""Bounded paired current-India fault study; configurable predeclared seeds.

Uses clean-run policy freeze without re-tuning under test faults. No older evidence modified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import pandas as pd
import yaml
from run_simulation import run_method
from run_evidence_improvement import policy_margin
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate, predictions_frame
from src.reserve.simulator import load_node, synthetic_demand
from src.reserve.online import schedule_online, evaluate_schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='outputs/evidence_improvement')
    ap.add_argument('--out', default='outputs/current_robustness')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = ap.parse_args()
    out, source = Path(args.out), Path(args.source)
    if (out/'failure_results.csv').exists():
        raise ValueError('Completed evidence cannot be overwritten; choose a new --out')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Seeds must be unique')
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((source/'config.yaml').read_text())
    freeze = json.loads((source/'policy_freeze.json').read_text())
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in
              [source, Path('outputs/mentor_validation')] for p in folder.glob('*') if p.is_file()}
    manifest = dict(seeds=args.seeds, scenarios=['stale_20_sites','dropout_50pct'],
                    clients=100, fault_sites=list(range(20)), clean_targets=True,
                    selection='Original clean-validation policy freeze; no retuning',
                    source_hashes=hashes, protocol='chronological, 12 rounds, 20-client cap',
                    interpretation='Paired run variability, not field reliability')
    (out/'preregistered.json').write_text(json.dumps(manifest, indent=2))
    policies = {p['name']:p for p in freeze['candidates']}
    rows = []
    for seed in args.seeds:
        rows.extend(run_seed(seed, cfg, freeze, policies, out))
        pd.DataFrame(rows).to_csv(out/'partial_results.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'failure_results.csv', index=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print('Paired multi-seed simulation; retained negative outcomes. Not service uptime.')


def run_seed(seed, cfg, freeze, policies, out):
    rows = []
    for scenario, faulty, dropout in [('stale_20_sites', list(range(20)), 0),
                                      ('dropout_50pct', [], .5)]:
        sc = copy.deepcopy(cfg); sc['federated']['dropout_rate'] = dropout
        clients, _ = prepare_clients(sc, 100, seed, faulty, 'stale' if faulty else None)
        paired = None
        for method in ['fedavg','reliability_fedavg','reliability_fedavg_event']:
            res = run_method(method, clients, sc, seed, faulty, verbose=False)
            m = evaluate(res, clients)
            node = load_node(predictions_frame(res, clients))
            node['demand_mw'] = synthetic_demand(node.time, node.capacity_mw.iloc[0], sc['reserve']['demand'], seed)
            identity = node[['time','actual_mw','demand_mw','capacity_mw','daytime','split']]
            if paired is None: paired = identity
            else: pd.testing.assert_frame_equal(paired, identity)
            name = freeze['selected'][method]
            d = schedule_online(node, policy_margin(node, policies[name]), sc['reserve'])
            rounds = res.extra['rounds']
            node.to_csv(out/f'node_{seed}_{scenario}_{method}.csv', index=False)
            rounds.to_csv(out/f'rounds_{seed}_{scenario}_{method}.csv', index=False)
            rows.append(dict(seed=seed, scenario=scenario, method=method,
                policy=name, healthy_rmse=m['healthy_rmse'], global_rmse=m['global_rmse'],
                comm_mb=m['total_comm_mb'], rejected_updates=int(rounds.n_rejected.sum()),
                rolled_back_rounds=int((rounds.rollback_reason!='').sum()),
                **evaluate_schedule(d, sc['reserve'], 'test')))
            print(f'{seed} {scenario} {method}: healthy RMSE {100*m["healthy_rmse"]:.3f}%', flush=True)
    return rows


if __name__ == '__main__': main()
