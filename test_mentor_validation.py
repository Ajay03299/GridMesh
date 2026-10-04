"""Regression checks for hourly input and strict paired reserve comparisons."""
import copy
import unittest
import numpy as np
import pandas as pd
from src.data.nasa_power import read_power
from src.data.virtual_sites import sample_site_params
from src.reserve.simulator import schedule_backup, simulate
from src.reliability.calibration import monitor_margin
from run_mentor_validation import india_config


class HourlyProtocolTests(unittest.TestCase):
    def setUp(self):
        self.cfg = india_config('unused.csv')
        self.rc = self.cfg['reserve']
        self.rc.update(grid_import_limit_mw=0, backup_power_limit_mw=2,
                       backup_energy_limit_mwh=3)
        self.node = pd.DataFrame({'time':pd.date_range('2024-01-01 08:30',periods=3,freq='h'),
            'split':'test','daytime':True,'capacity_mw':4., 'demand_mw':2.,
            'forecast_mw':0.,'actual_mw':0.,'block':0})

    def test_hourly_energy_and_objective(self):
        d,s = simulate(self.node,np.zeros(3),self.rc,'test')
        self.assertAlmostEqual(d.scheduled_backup_mw.sum(),3.)
        self.assertAlmostEqual(s['reserve_energy_mwh'],3.)
        self.assertAlmostEqual(s['shortfall_energy_mwh'],3.)
        self.assertAlmostEqual(s['planning_objective_score'],63.)
        self.assertAlmostEqual(s['total_cost'],63.)
        self.assertTrue((d.scheduled_backup_mw <=2).all())

    def test_half_hour_budget_uses_real_interval(self):
        rc = copy.deepcopy(self.rc);rc['step_hours']=.5
        d,s = simulate(self.node,np.zeros(3),rc,'test')
        self.assertAlmostEqual(s['reserve_energy_mwh'],3.)
        self.assertAlmostEqual(s['shortfall_energy_mwh'],0.)

    def test_same_grid_for_different_forecasts(self):
        self.rc['grid_import_limit_mw']=1.
        a = schedule_backup(self.node,np.zeros(3),self.rc)
        node = self.node.copy();node['forecast_mw']=1.
        b = schedule_backup(node,np.zeros(3),self.rc)
        np.testing.assert_array_equal(a.grid_import_mw,b.grid_import_mw)
        np.testing.assert_array_equal(a.planning_unserved_cost_per_mwh,b.planning_unserved_cost_per_mwh)

    def test_invalid_margin_and_duration(self):
        for margin in ([-1,0,0],[float('nan'),0,0],[1,2]):
            with self.assertRaises(ValueError):schedule_backup(self.node,margin,self.rc)
        for dt in (0,-1,float('nan')):
            rc = dict(self.rc,step_hours=dt)
            with self.assertRaises(ValueError):schedule_backup(self.node,[0,0,0],rc)

    def test_rooftop_capacity_not_rounded_to_zero(self):
        capacities=[sample_site_params(i,self.cfg['virtual_sites'],42).capacity_mw for i in range(100)]
        self.assertTrue(all(.002 <= x <=.008 for x in capacities))
        self.assertGreater(len(set(capacities)),90)

    def test_hourly_calibration_age(self):
        h=monitor_margin(self.node,[0,0,0],.05,6,1,1,[1,1,1],step_minutes=60)
        self.assertEqual(h.calibration_age_minutes.iloc[1],0.)

    def test_indian_provenance_and_no_interpolation(self):
        frame,p=read_power('data/raw/nasa_power_bengaluru_2024.json')
        self.assertEqual(len(frame),8784)
        self.assertTrue((frame.timestamp.diff().dropna() == pd.Timedelta(hours=1)).all())
        self.assertEqual(str(frame.timestamp.iloc[0]),'2024-01-01 05:30:00')
        self.assertEqual(p['step_hours'],1)
        self.assertEqual(p['sha256'],'c6c267e012d9da7c00801437e78c9e07e551137bfbb51f326a81c314a756d07c')
        self.assertTrue(np.isfinite(frame.GHI).all())


if __name__ == '__main__':unittest.main()
