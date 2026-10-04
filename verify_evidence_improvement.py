"""Replay every frozen policy decision/metric with current code; no retraining/tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from run_evidence_improvement import policy_margin, select_policy
from src.reserve.online import schedule_online, evaluate_schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='outputs/evidence_improvement')
    args = ap.parse_args(); out = Path(args.out)
    cfg = yaml.safe_load((out/'config.yaml').read_text())
    spec = yaml.safe_load((out/'experiment.yaml').read_text())
    frozen = json.loads((out/'policy_freeze.json').read_text())
    validation = pd.read_csv(out/'validation_candidates.csv')
    test = pd.read_csv(out/'test_results.csv')
    policies = {p['name']:p for p in spec['policies']}
    count = 0
    for seed in spec['seeds']:
        for method in spec['methods']:
            n = pd.read_csv(out/f'node_{seed}_{method}.csv', parse_dates=['time'])
            v = n[n.split=='val'].reset_index(drop=True)
            for p in spec['policies']:
                actual = evaluate_schedule(schedule_online(v, policy_margin(v,p), cfg['reserve']), cfg['reserve'], 'val')
                recorded = validation[(validation.seed==seed)&(validation.method==method)&(validation.policy==p['name'])].iloc[0]
                for key,value in actual.items(): np.testing.assert_allclose(value, recorded[key], atol=1e-9, rtol=1e-9)
            for row in test[(test.seed==seed)&(test.method==method)].itertuples():
                p = policies[row.policy]
                actual = evaluate_schedule(schedule_online(n, policy_margin(n,p), cfg['reserve']), cfg['reserve'], 'test')
                for key,value in actual.items(): np.testing.assert_allclose(value, getattr(row,key), atol=1e-9, rtol=1e-9)
                count += 1
    for method in spec['methods']:
        assert select_policy(validation[validation.method==method]) == frozen['selected'][method]
    result = dict(policy_test_rows_replayed=count, validation_rows_replayed=len(validation),
                  frozen_selection_matches=True, test_results_sha256=hashlib.sha256((out/'test_results.csv').read_bytes()).hexdigest())
    (out/'replay_verification.json').write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == '__main__': main()
