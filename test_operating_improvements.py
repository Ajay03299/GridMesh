import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from build_advisory_demo import SCENARIOS, scenario_demo
from run_operating_sensitivity import scenario_inputs
from src.reserve.operating import advisory_inputs, joint_error_margin
from src.reserve.online import schedule_online
from src.evaluation.pilot_costs import full_service

class OperatingTests(unittest.TestCase):
    def test_all_five_scenarios_have_finite_advice(self):
        for name in SCENARIOS:
            d,m=scenario_demo(name)
            self.assertTrue(np.isfinite(d.scheduled_backup_mw).all())
            self.assertTrue((d.scheduled_backup_mw>=0).all())
            self.assertTrue((d.remaining_energy_before_mwh>=-1e-9).all())
            self.assertGreaterEqual(m['ens_mwh'],0)
            self.assertTrue(d.reason.str.len().gt(0).all())
        d,m=scenario_demo('insufficient')
        self.assertGreater(m['ens_mwh'],0)
        self.assertTrue(d.operator_attention.any())

    def test_missing_stale_and_degraded_are_bounds_not_forecasts(self):
        d,_=scenario_demo('stale_missing')
        self.assertEqual(d.forecast_source.iloc[1],'zero_solar_safety_bound')
        self.assertEqual(d.forecast_source.iloc[2],'zero_solar_safety_bound')
        self.assertEqual(d.forecast_mw.iloc[2],0)
        self.assertTrue(d.operator_attention.iloc[2])
        d,_=scenario_demo('dropout')
        self.assertTrue((d.forecast_source.iloc[1:4]=='zero_solar_safety_bound').all())

    def test_invalid_age_and_timestamp(self):
        d,_=scenario_demo('normal')
        d.loc[0,'forecast_age_minutes']=np.nan
        self.assertEqual(advisory_inputs(d).forecast_mw.iloc[0],0)
        cfg=dict(grid_dispatch_mode='fixed_availability',grid_import_limit_mw=.05,
            backup_power_limit_mw=.03,backup_energy_limit_mwh=.06,step_hours=1.,
            cost_reserve_per_mwh=1.,cost_shortfall_per_mwh=20.)
        d.loc[0,'time']=pd.NaT
        with self.assertRaises(ValueError):schedule_online(d,np.zeros(len(d)),cfg)

    def test_joint_margin_never_sees_current_or_future_truth(self):
        d,_=scenario_demo('normal')
        a=joint_error_margin(d,.9,window=3,min_periods=1)
        d.loc[2:,'actual_demand_mw']=9
        d.loc[2:,'actual_grid_mw']=0
        d.loc[2:,'actual_mw']=0
        b=joint_error_margin(d,.9,window=3,min_periods=1)
        np.testing.assert_array_equal(a[:3],b[:3])

    def test_stress_events_are_daytime_and_deterministic(self):
        node=pd.DataFrame(dict(time=pd.date_range('2024-01-01',periods=48,freq='h'),
            daytime=[False]*8+[True]*12+[False]*12+[True]*12+[False]*4,
            demand_mw=.08,capacity_mw=.1,actual_mw=.02,forecast_mw=.025,split='test'))
        cfg=yaml.safe_load(Path('outputs/evidence_improvement/config.yaml').read_text())['reserve']
        scenario={'event_every':8,'delivery_loss_fraction':1}
        a,_=scenario_inputs(node,cfg,scenario,42)
        b,_=scenario_inputs(node,cfg,scenario,42)
        pd.testing.assert_frame_equal(a,b)
        affected=a.actual_backup_fraction.eq(0)
        self.assertTrue(a.loc[affected,'daytime'].all())
        self.assertEqual(affected.sum(),int(np.ceil(a.daytime.sum()/8)))
        for invalid in ({'delivery_loss_fraction':2},{'event_every':0},{'demand_std':-1}):
            with self.assertRaises(ValueError):scenario_inputs(node,cfg,invalid,42)

    def test_costs_missing_tax_and_capital_recovery(self):
        cfg=yaml.safe_load(Path('configs/full_service_costs.yaml').read_text())
        r=full_service(cfg)[0]
        self.assertFalse(r['complete'])
        self.assertIsNone(r['break_even_monthly_fee_inr'])
        self.assertIn('replacement_provision',full_service(cfg)[-1]['unknown_items'])
        self.assertNotIn('financing_payment',full_service(cfg)[-1]['unknown_items'])
        cfg['capital_recovery']='financing'
        self.assertIn('financing_payment',full_service(cfg)[-1]['unknown_items'])
        cfg['items'][0]['inr']=float('nan')
        with self.assertRaises(ValueError):full_service(cfg)

if __name__=='__main__':unittest.main()
