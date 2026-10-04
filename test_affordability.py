import unittest
from copy import deepcopy
from pathlib import Path
import yaml
from src.evaluation.pilot_costs import affordability_plan, scenarios


class AffordabilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg=yaml.safe_load(Path('configs/affordability_costs.yaml').read_text())

    def item(self,name):
        return next(i for i in self.cfg['items'] if i['name']==name)

    def test_item_totals_taxes_and_setup(self):
        r=affordability_plan(self.cfg)
        # 27USD x 95 x 1.18 + 499 x 1.18 + paid operator + two distinct contractors
        cash=27*95*1.18+499*1.18+15*250+4*750*1.18+2*750*1.18
        upfront=20*750*1.18+4*250+2500*1.18
        self.assertAlmostEqual(r['known_monthly_incremental_cash_inr'],cash)
        self.assertAlmostEqual(r['known_upfront_incremental_inr'],upfront)
        self.assertAlmostEqual(r['setup_recovery_monthly_inr'],upfront/36)
        self.assertAlmostEqual(r['covered_service_monthly_inr'],(cash+upfront/36)*1.15)
        self.cfg['setup_recovery_months']=12
        self.assertAlmostEqual(affordability_plan(self.cfg)['setup_recovery_monthly_inr'],upfront/12)

    def test_unit_conversion_and_no_scheduled_energy_price(self):
        e=self.item('consumed_backup_energy');e.update(quantity=.6,rate=10,tax=0)
        r=affordability_plan(self.cfg)
        self.assertEqual(next(d for d in r['details'] if d['name']==e['name'])['gross_inr'],6000)
        self.assertIsNone(r['full_incremental_monthly_inr'])
        self.assertIsNone(r['operating_change_monthly_inr'])

    def test_participation_and_shortfall(self):
        a=affordability_plan(self.cfg,paying_homes=100)
        b=affordability_plan(self.cfg,paying_homes=50)
        self.assertAlmostEqual(b['covered_cost_recovery_per_home_inr'],2*a['covered_cost_recovery_per_home_inr'])
        self.cfg['funding']['proposed_household_payment_inr']=a['covered_cost_recovery_per_home_inr']
        self.assertAlmostEqual(affordability_plan(self.cfg,paying_homes=50)['covered_shortfall_inr'],a['covered_service_monthly_inr']/2)
        for homes in (0,101,True,1.5):
            with self.assertRaises(ValueError):affordability_plan(self.cfg,paying_homes=homes)

    def test_missing_values_never_zero_total(self):
        r=affordability_plan(self.cfg)
        self.assertFalse(r['incremental_complete']);self.assertIsNone(r['total_operating_monthly_inr'])
        self.item('cloud_compute')['tax']=None
        r=affordability_plan(self.cfg)
        self.assertIn('cloud_compute: tax',r['unknown_items']['incremental'])
        self.assertIsNone(next(d for d in r['details'] if d['name']=='cloud_compute')['gross_inr'])
        new=affordability_plan(self.cfg,asset_scenario='new_equipment')
        self.assertIsNone(new['initial_capital_cash_inr'])
        self.assertFalse(new['initial_capital_complete'])

    def test_capital_recovery_finance_are_alternatives(self):
        self.item('new_backup_equipment').update(rate=60000,tax=0)
        self.item('new_backup_financing').update(rate=1200,tax=0)
        a=affordability_plan(self.cfg,asset_scenario='new_equipment')
        self.assertNotIn('new_backup_financing',[d['name'] for d in a['details']])
        self.assertEqual(next(d for d in a['details'] if d['name']=='new_backup_equipment')['recovery_monthly_inr'],1000)
        self.cfg['capital_recovery']='financing'
        b=affordability_plan(self.cfg,asset_scenario='new_equipment')
        self.assertEqual(next(d for d in b['details'] if d['name']=='new_backup_equipment')['recovery_monthly_inr'],0)
        self.assertAlmostEqual(b['covered_total_operating_monthly_inr']-a['covered_total_operating_monthly_inr'],200)
        original=yaml.safe_load(Path('configs/pilot_costs.yaml').read_text())
        for key,value in [('new_asset_capex',60000),('new_asset_life_months',60),('new_asset_financing_monthly',1200)]:original[key]['value']=value
        with self.assertRaises(ValueError):scenarios(original)

    def test_conditional_funding_requires_named_payer(self):
        base=affordability_plan(self.cfg)
        self.cfg['funding'].update(sponsor_monthly_inr=1000,sponsor_name='Hypothetical existing operator')
        ordinary=affordability_plan(self.cfg)
        self.assertEqual(ordinary['sponsor_applied_inr'],0)
        conditional=affordability_plan(self.cfg,conditional_funding=True)
        self.assertTrue(conditional['conditional_funding'])
        self.assertAlmostEqual(conditional['covered_cost_recovery_per_home_inr'],base['covered_cost_recovery_per_home_inr']-10)
        self.cfg['funding']['sponsor_name']=None
        with self.assertRaises(ValueError):affordability_plan(self.cfg,conditional_funding=True)

    def test_configuration_updates_outputs_and_low_high_order(self):
        costs=[affordability_plan(self.cfg,band=b)['covered_service_monthly_inr'] for b in ('low','base','high')]
        self.assertLess(costs[0],costs[1]);self.assertLess(costs[1],costs[2])
        old=costs[1];self.item('operator_review')['quantity']['base']+=1
        self.assertAlmostEqual(affordability_plan(self.cfg)['covered_service_monthly_inr']-old,250*1.15)
        self.item('operator_review')['rate']['base']=float('nan')
        with self.assertRaises(ValueError):affordability_plan(self.cfg)

    def test_completed_scope_and_delta_do_not_double_count(self):
        for i in self.cfg['items']:
            if i['rate'] is None:i['rate']=100
            if i['quantity'] is None:i['quantity']=1
            if i['tax'] is None:i['tax']=0
        self.cfg['operating_change_monthly_inr']=10
        a=affordability_plan(self.cfg)
        self.assertTrue(a['incremental_complete']);self.assertTrue(a['total_complete'])
        self.assertAlmostEqual(a['full_incremental_monthly_inr'],a['covered_service_monthly_inr']+10)
        self.cfg['operating_change_monthly_inr']=20
        b=affordability_plan(self.cfg)
        self.assertEqual(a['total_operating_monthly_inr'],b['total_operating_monthly_inr'])
        self.assertFalse(b['price_validated'])


if __name__=='__main__':unittest.main()
