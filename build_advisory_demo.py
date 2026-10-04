"""No training/network required: deterministic cloud shock + exhausted backup advisory."""
import json
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
from src.reserve.online import schedule_online, advice, evaluate_schedule
from src.reserve.operating import advisory_inputs, operating_metrics
from src.evaluation.pilot_costs import scenarios


def demo():
    targets = pd.date_range('2024-10-15 09:30', periods=7, freq='h')
    n = pd.DataFrame(dict(time=targets, split='test', daytime=True,
        capacity_mw=.5, demand_mw=.5, forecast_mw=[.25,.18,.07,.06,.08,.12,.23],
        actual_mw=[.24,.12,.02,.03,.07,.10,.21],
        forecast_age_minutes=[0,0,0,120,0,0,0], forecast_warning=[False,False,False,True,False,False,False]))
    cfg = dict(grid_dispatch_mode='fixed_availability', grid_import_limit_mw=.3,
        backup_power_limit_mw=.1, backup_energy_limit_mwh=.20, step_hours=1.,
        cost_reserve_per_mwh=1., cost_shortfall_per_mwh=20., peak_priority_weight=0.)
    d = schedule_online(n, [.02,.02,.03,.04,.03,.02,.02], cfg)
    d['issued_at'] = d.time-pd.Timedelta(hours=1)
    d['forecast_horizon_minutes'] = 60
    d['reason'] = [advice(r)['reason'] for r in d.to_dict('records')]
    d['operator_attention'] = [advice(r)['operator_attention'] for r in d.to_dict('records')]
    return d, evaluate_schedule(d, cfg, 'test')


SCENARIOS = {
    'normal': 'Normal operation',
    'cloud': 'Cloud-driven shortfall',
    'insufficient': 'Insufficient backup',
    'stale_missing': 'Stale / missing solar input',
    'dropout': 'Client dropout / degraded-site warning',
}


def scenario_demo(name):
    """Deterministic, 100-home assumed pilot scale; independent of research results."""
    if name not in SCENARIOS:
        raise ValueError('Unknown demo scenario')
    times=pd.date_range('2024-10-15 10:30',periods=5,freq='h')
    cloud=name in ('cloud','insufficient')
    n=pd.DataFrame(dict(time=times,split='test',daytime=True,capacity_mw=.1,
        demand_mw=.08,forecast_mw=.015 if cloud else .04,actual_mw=.01 if cloud else .035,
        forecast_age_minutes=0.,forecast_warning=False))
    if name=='stale_missing':
        n.loc[1,'forecast_age_minutes']=120
        n.loc[2,'forecast_mw']=np.nan
    if name=='dropout':
        n.loc[1:3,'forecast_warning']=True
    n['actual_demand_mw']=n.demand_mw
    n['planned_grid_mw']=.05
    n['actual_grid_mw']=.05
    n['actual_backup_fraction']=1.
    cfg=dict(grid_dispatch_mode='fixed_availability',grid_import_limit_mw=.05,
        backup_power_limit_mw=.005 if name=='insufficient' else .03,
        backup_energy_limit_mwh=.01 if name=='insufficient' else .06,
        step_hours=1.,cost_reserve_per_mwh=1.,cost_shortfall_per_mwh=20.,peak_priority_weight=0.)
    prepared=advisory_inputs(n)
    d=schedule_online(prepared,np.full(len(n),.01),cfg)
    d['issued_at']=d.time-pd.Timedelta(hours=1)
    d['forecast_horizon_minutes']=60
    d['reason']=[advice(r)['reason'] for r in d.to_dict('records')]
    d['operator_attention']=[advice(r)['operator_attention'] for r in d.to_dict('records')]
    d['actual_delivered_backup_mw']=np.minimum(d.scheduled_backup_mw,(d.actual_demand_mw-d.actual_grid_mw-d.actual_mw).clip(lower=0))
    d['realized_unserved_mwh']=(d.actual_demand_mw-d.actual_grid_mw-d.actual_mw-d.scheduled_backup_mw*d.actual_backup_fraction).clip(lower=0)*d.step_hours
    d['participating_clients']=80 if name=='dropout' else 100
    return d,operating_metrics(d,cfg,'test')


def main():
    out = Path('outputs/evidence_improvement_demo'); out.mkdir(parents=True, exist_ok=True)
    d, metrics = demo()
    d.to_csv(out/'advisory.csv', index=False)
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2))
    costs = yaml.safe_load(Path('configs/pilot_costs.yaml').read_text())
    pd.DataFrame(scenarios(costs)).to_csv(out/'pilot_cost_sensitivity.csv', index=False)
    suite=Path('outputs/operator_scenarios'); suite.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for name in SCENARIOS:
        rows,outcome=scenario_demo(name)
        rows.to_csv(suite/f'{name}.csv',index=False)
        summaries.append(dict(scenario=name,**outcome))
    pd.DataFrame(summaries).to_csv(suite/'summary.csv',index=False)
    print(d[['time','expected_gap_mw','required_backup_mw','scheduled_backup_mw','planned_gap_mw','remaining_energy_before_mwh','reason']].to_string(index=False))
    print('Synthetic advisory only. Generic backup; no physical equipment/control. INR estimates incomplete.')


if __name__ == '__main__':
    main()
