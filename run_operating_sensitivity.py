"""Frozen-forecast, paired operating stresses; never overwrite prior evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from run_evidence_improvement import policy_margin, select_policy
from src.reserve.simulator import _asset_limits
from src.reserve.online import schedule_online
from src.reserve.operating import advisory_inputs, joint_error_margin, operating_metrics


def scenario_inputs(node, cfg, scenario, seed):
    for name in ('grid_loss_fraction','delivery_loss_fraction','solar_loss_fraction'):
        value=scenario.get(name,0)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be a finite fraction between0 and1')
    for name in ('known_power_factor','known_energy_factor','demand_std'):
        value=scenario.get(name,0 if name=='demand_std' else 1)
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not np.isfinite(scenario.get('demand_bias',0)):
        raise ValueError('demand_bias must be finite')
    for name in ('event_every','stale_every'):
        value=scenario.get(name,1)
        if isinstance(value,bool) or not isinstance(value,int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    d=node.copy().reset_index(drop=True)
    grid,power,energy=_asset_limits(d,cfg)
    operating_cfg=cfg | {'grid_import_limit_mw':grid,
        'backup_power_limit_mw':power*scenario.get('known_power_factor',1),
        'backup_energy_limit_mwh':energy*scenario.get('known_energy_factor',1)}
    rng=np.random.default_rng([seed,902])
    error=rng.normal(scenario.get('demand_bias',0),scenario.get('demand_std',0),len(d))
    d['actual_demand_mw']=d.demand_mw*(1+error).clip(min=0)
    d['planned_grid_mw']=np.minimum(grid,d.demand_mw)
    # Stresses target evaluated daytime intervals, never choose by observed errors.
    daytime_rank=d.daytime.astype(int).cumsum().to_numpy()-1
    event=d.daytime.to_numpy() & ((daytime_rank % scenario.get('event_every',6))==0)
    d['actual_grid_mw']=np.minimum(d.planned_grid_mw*(1-event*scenario.get('grid_loss_fraction',0)),d.actual_demand_mw)
    d['actual_backup_fraction']=1-event*scenario.get('delivery_loss_fraction',0)
    d['actual_mw']=d.actual_mw*(1-event*scenario.get('solar_loss_fraction',0))
    stale=(np.arange(len(d)) % scenario.get('stale_every',len(d)+1))==1 if 'stale_every' in scenario else np.zeros(len(d),bool)
    d.loc[stale,'forecast_mw']=d.forecast_mw.shift(1).loc[stale]
    d['forecast_age_minutes']=np.where(stale,120,0)
    d['forecast_warning']=stale
    return d,operating_cfg


def schedule_policy(node,cfg,policy,joint_cfg):
    if policy.startswith('joint_'):
        d=advisory_inputs(node)
        margin=joint_error_margin(d,float(policy.split('_')[1]),joint_cfg['window'],joint_cfg['min_periods'])
    else:
        d=node.copy()
        spec={'family':'nsigma','delta':.05,'window':6} if policy=='nominal_05' else {'family':'fixed','fraction':float(policy.split('_')[1].removesuffix('pct'))/100}
        margin=policy_margin(d,spec)
    return schedule_online(d,margin,cfg)


def choose_guard(validation,target):
    base=validation[validation.policy=='fixed_20pct'].set_index('seed')
    eligible=[]
    for name,rows in validation[validation.policy.str.startswith('joint_')].groupby('policy'):
        rows=rows.set_index('seed').reindex(base.index)
        if ((rows.availability_pct>=target)&(rows.ens_mwh<=base.ens_mwh+1e-9)).all():
            eligible.append((rows.backup_mwh.mean(),name))
    return min(eligible)[1] if eligible else 'fixed_20pct'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='outputs/evidence_improvement')
    ap.add_argument('--config',default='configs/operating_sensitivity.yaml')
    ap.add_argument('--out',default='outputs/operating_sensitivity_v1')
    args=ap.parse_args(); source,out=Path(args.source),Path(args.out)
    if (out/'preregistered.json').exists(): raise ValueError('Choose a new output directory; do not retune completed evidence')
    spec=yaml.safe_load(Path(args.config).read_text()); cfg=yaml.safe_load((source/'config.yaml').read_text())['reserve']
    if not 0 <= spec['target_validation_availability_pct'] <= 100:
        raise ValueError('Availability target must be between0 and100 percent')
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in [source,Path('outputs/mentor_validation')] for p in folder.glob('*') if p.is_file()}
    out.mkdir(parents=True,exist_ok=True)
    candidates=['fixed_20pct','fixed_10pct','fixed_15pct','nominal_05']+[f'joint_{q}' for q in spec['joint_margin']['quantiles']]
    prereg=dict(config=spec,candidates=candidates,source_hashes=hashes,
        assumptions='Engineering stresses, not empirically fitted distributions. Outcomes of interval t observable before t+1; generic backup.',
        selection='Validation only: joint target >=99% in each seed and ENS <= fixed20; lowest backup, else fixed fallback.',
        guard='Experimental joint net-gap error quantile; existing nominal default unchanged.')
    (out/'preregistered.json').write_text(json.dumps(prereg,indent=2))
    val=[]; cache={}
    for name,scenario in spec['scenarios'].items():
        for seed in spec['seeds']:
            node=pd.read_csv(source/f'node_{seed}_{spec["method"]}.csv',parse_dates=['time'])
            d,scfg=scenario_inputs(node,cfg,scenario,seed); cache[name,seed]=d,scfg
            v=d[d.split=='val'].reset_index(drop=True)
            for p in candidates:
                s=schedule_policy(v,scfg,p,spec['joint_margin'])
                val.append(dict(scenario=name,seed=seed,policy=p,**operating_metrics(s,scfg,'val')))
    validation=pd.DataFrame(val); validation.to_csv(out/'validation.csv',index=False)
    selected={}
    for name in spec['scenarios']:
        v=validation[validation.scenario==name]
        selected[name]=dict(guard=choose_guard(v,spec['target_validation_availability_pct']),
            simple=select_policy(v[v.policy.isin(['fixed_20pct','fixed_10pct','fixed_15pct'])]))
    freeze=dict(selected=selected,test_used_for_selection=False)
    (out/'policy_freeze.json').write_text(json.dumps(freeze,indent=2))
    rows=[]
    for (name,seed),(d,scfg) in cache.items():
        roles={'fixed':'fixed_20pct','existing_nominal':'nominal_05','validation_simple':selected[name]['simple'],'validation_guard':selected[name]['guard']}
        for role,p in roles.items():
            s=schedule_policy(d,scfg,p,spec['joint_margin'])
            metrics=operating_metrics(s,scfg,'test')
            rows.append(dict(scenario=name,seed=seed,role=role,policy=p,**metrics))
            if seed==spec['seeds'][0]:s.to_csv(out/f'schedule_{name}_{role}.csv',index=False)
    results=pd.DataFrame(rows); results.to_csv(out/'test_results.csv',index=False)
    summary=results.groupby(['scenario','role'])[['availability_pct','ens_mwh','backup_mwh','cost_score']].mean()
    summary.to_csv(out/'summary.csv')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()),'Frozen evidence changed'
    (out/'verified.json').write_text(json.dumps(dict(source_unchanged=True,test_rows=len(rows),validation_rows=len(val),
        freeze_sha256=hashlib.sha256((out/'policy_freeze.json').read_bytes()).hexdigest()),indent=2))
    print(summary.to_string())


if __name__=='__main__':main()
