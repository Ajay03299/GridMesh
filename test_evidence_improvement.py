import copy
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from build_advisory_demo import demo
from run_evidence_improvement import policy_margin, select_policy
from src.reserve.online import schedule_online, evaluate_schedule, advice
from src.evaluation.pilot_costs import scenarios
from src.reliability.safety import safe_forecast


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.node = pd.DataFrame(dict(time=pd.date_range('2024-01-01 10:00', periods=4, freq='h'),
            split='test', daytime=True, forecast_mw=.1, actual_mw=.1, demand_mw=.6, capacity_mw=1., block=0))
        self.cfg = dict(grid_dispatch_mode='fixed_availability', grid_import_limit_mw=.3,
            backup_power_limit_mw=.15, backup_energy_limit_mwh=.25, step_hours=1.,
            cost_reserve_per_mwh=1., cost_shortfall_per_mwh=20.)

    def test_power_energy_and_planned_vs_actual(self):
        d = schedule_online(self.node, [0]*4, self.cfg)
        self.assertAlmostEqual(d.scheduled_backup_mw.sum(), .25)
        self.assertLessEqual(d.scheduled_backup_mw.max(), .15)
        m = evaluate_schedule(d, self.cfg, 'test')
        self.assertAlmostEqual(m['ens_mwh'], .55)
        high_solar = d.assign(actual_mw=.6)
        self.assertEqual(evaluate_schedule(high_solar, self.cfg, 'test')['ens_mwh'], 0)
        self.assertGreater(m['planned_uncovered_mwh'], 0)

    def test_future_inputs_and_actual_cannot_change_earlier_decision(self):
        a = schedule_online(self.node, [0]*4, self.cfg)
        changed = self.node.copy()
        changed.loc[2:, ['forecast_mw','demand_mw']] = [.8, 1.]
        changed['actual_mw'] = 1000
        b = schedule_online(changed, [0,0,1,1], self.cfg)
        np.testing.assert_array_equal(a.scheduled_backup_mw.iloc[:2], b.scheduled_backup_mw.iloc[:2])

    def test_margin_does_not_see_future_actual(self):
        n = pd.concat([self.node]*4, ignore_index=True)
        n.time = pd.date_range('2024-01-01', periods=len(n), freq='h')
        p = dict(family='nsigma', delta=.05, window=6)
        a = policy_margin(n, p)
        n.loc[10:, 'actual_mw'] = 20
        b = policy_margin(n, p)
        np.testing.assert_array_equal(a[:11], b[:11])

    def test_zero_and_stale_explanation(self):
        d, _ = demo()
        zero = schedule_online(self.node.assign(forecast_mw=.8), [0]*4, self.cfg)
        self.assertEqual(zero.scheduled_backup_mw.iloc[0], 0)
        self.assertIn('No forecast gap', advice(zero.iloc[0])['reason'])
        self.assertTrue(advice(d.iloc[3])['operator_attention'])
        self.assertEqual(d.scheduled_backup_mw.iloc[4], 0)
        self.assertIn('budget', advice(d.iloc[4])['reason'])

    def test_invalid_input_rejected(self):
        for v in (float('nan'), -1, float('inf')):
            n = self.node.copy(); n.loc[0, 'forecast_mw'] = v
            with self.assertRaises(ValueError): schedule_online(n, [0]*4, self.cfg)
        cfg = self.cfg | {'backup_energy_limit_mwh':float('nan')}
        with self.assertRaises(ValueError): schedule_online(self.node, [0]*4, cfg)
        with self.assertRaises(ValueError): schedule_online(self.node.drop(columns='capacity_mw'), [0]*4, self.cfg)
        with self.assertRaises(ValueError): schedule_online(self.node.iloc[::-1], [0]*4, self.cfg)

    def test_selection_rejects_reliability_loss_and_has_fallback(self):
        rows = pd.DataFrame([dict(seed=s, policy=p, availability_pct=a, ens_mwh=e, backup_mwh=b, cost_score=b+20*e)
            for s in (42,43) for p,a,e,b in [('fixed_20pct',98,1,10), ('unsafe',97,2,5), ('safe',98,1,8)]])
        self.assertEqual(select_policy(rows), 'safe')
        self.assertEqual(select_policy(rows[rows.policy != 'safe']), 'fixed_20pct')
        with self.assertRaises(ValueError): select_policy(rows[rows.policy != 'fixed_20pct'])
        with self.assertRaises(ValueError): select_policy(pd.concat([rows, rows.iloc[:1]]))

    def test_invalid_policy_configuration(self):
        for p in [dict(family='unknown'), dict(family='fixed', fraction=-1),
                  dict(family='nsigma', delta=0, window=6)]:
            with self.assertRaises(ValueError): policy_margin(self.node, p)

    def test_unknown_costs_are_not_complete_prices(self):
        cfg = yaml.safe_load(Path('configs/pilot_costs.yaml').read_text())
        rows = scenarios(cfg)
        r = next(r for r in rows if r['paying_homes']==100 and r['assumed_additional_monthly_inr']==0)
        # Unsupported historical 4650 is no longer an active pricing input.
        self.assertEqual(r['known_subtotal_per_paying_home_inr'], 0)
        self.assertIn('software_hosting_monthly', r['unknown_items'])
        self.assertFalse(r['complete'])
        self.assertIn('operator_labour_monthly', r['unknown_items'])
        bad = copy.deepcopy(cfg); bad['participation'] = [0]
        with self.assertRaises(ValueError): scenarios(bad)

    def test_runtime_failure_falls_back_without_claiming_uptime(self):
        p, health = safe_forecast(np.array([np.nan]), None, None, np.array([.2]))
        self.assertEqual(health['source'], 'smart_persistence')
        self.assertTrue(health['operator_attention'])
        np.testing.assert_array_equal(p, [.2])


if __name__ == '__main__': unittest.main()
