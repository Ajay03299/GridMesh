"""Validate completed evidence and produce a schedule plot plus reproducibility manifest."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import scipy
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data.nasa_power import read_power
from src.evaluation.serialization import dumps
from run_mentor_validation import check_schedule


def main():
    out=Path('outputs/mentor_validation')
    fm=pd.read_csv(out/'forecasting_metrics.csv')
    sites=pd.read_csv(out/'sitewise_metrics.csv')
    ops=pd.read_csv(out/'model_to_operations_comparison.csv')
    schedule=pd.read_csv(out/'reserve_schedule_comparison.csv',parse_dates=['time'])
    assert len(fm)==15 and fm.seed.nunique()==5 and set(fm.model)=={'mlp','gru','lstm'}
    assert len(sites)==1500 and sites.site.nunique()==100
    assert len(ops)==30 and set(ops.policy)=={'fixed_20pct','nsigma_d0.05'}
    assert np.isfinite(fm[['global_rmse','global_mae']]).all().all()
    for _,group in schedule.groupby(['seed','model','policy']):
        check_schedule(group.reset_index(drop=True))
        ev=group[(group.split=='test')&group.daytime]
        summary=ops[(ops.seed==group.seed.iloc[0])&(ops.model==group.model.iloc[0])&
                    (ops.policy==group.policy.iloc[0])].iloc[0]
        assert abs(summary.reserve_energy_mwh-(ev.scheduled_backup_mw*ev.step_hours).sum())<1e-8
        assert abs(summary.shortfall_energy_mwh-(ev.shortfall_mw*ev.step_hours).sum())<1e-8
    _,provenance=read_power('data/raw/nasa_power_bengaluru_2024.json')
    (out/'provenance.json').write_text(dumps(provenance,indent=2))
    representative=pd.read_csv(out/'representative_schedule.csv',parse_dates=['time'])
    day=representative[representative.daytime]
    # Scientific data visualization, never a decorative/illustrative asset.
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True)
    palette={'demand_mw':'#68645E','forecast_mw':'#174A72','expected_gap_mw':'#9B661B',
        'uncertainty_margin_mw':'#C59650','scheduled_backup_mw':'#23725C','shortfall_mw':'#080808'}
    groups=[['demand_mw','forecast_mw'],['expected_gap_mw','uncertainty_margin_mw',
        'scheduled_backup_mw'],['shortfall_mw']]
    labels={'demand_mw':'Synthetic demand','forecast_mw':'Solar forecast','expected_gap_mw':'Expected gap',
        'uncertainty_margin_mw':'Uncertainty margin','scheduled_backup_mw':'Scheduled backup',
        'shortfall_mw':'Realized unmet demand'}
    for ax,cols in zip(axes,groups):
        for col in cols:ax.plot(day.time,day[col]*1000,color=palette[col],label=labels[col],lw=2)
        ax.set_ylabel('kW');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=10)
    axes[1].axhline(day.backup_power_limit_mw.iloc[0]*1000,color='#68645E',ls='--',label='Backup power cap')
    axes[1].legend(loc='upper left',fontsize=10)
    axes[-1].set_xlabel('India Standard Time')
    fig.suptitle(f"MLP / nominal n-sigma / seed 42 / {day.time.dt.date.iloc[0]}\n"
        f"Daily backup: {day.scheduled_backup_mw.sum():.3f} MWh, budget {day.backup_energy_limit_mwh.iloc[0]:.3f} MWh\n"
        'Retrospective replay. Modeled PV, synthetic load, assumed assets.',fontsize=13)
    fig.tight_layout();fig.savefig(out/'representative_schedule.png',dpi=180);plt.close(fig)
    chart={"date":str(day.time.dt.date.iloc[0]),"categories":day.time.dt.strftime('%H:%M').tolist(),
        "series":[{'name':labels[col],'values':(day[col]*1000).tolist(),'color':palette[col]}
                  for col in palette],
        'daily_budget_mwh':float(day.backup_energy_limit_mwh.iloc[0]),
        'daily_scheduled_mwh':float(day.scheduled_backup_mw.sum()),
        'power_cap_kw':float(day.backup_power_limit_mw.iloc[0]*1000)}
    chart['required_kw']=(day.required_backup_mw*1000).tolist()
    chart['scheduled_kw']=(day.scheduled_backup_mw*1000).tolist()
    chart['planned_uncovered_kw']=(day.planned_gap_mw*1000).tolist()
    (out/'schedule_chart.json').write_text(dumps(chart,indent=2))
    if (out/'legacy_verification.json').exists():
        legacy=json.loads((out/'legacy_verification.json').read_text())
        summary=json.loads((out/'summary_metrics.json').read_text());summary['legacy_reproduction']=legacy
        (out/'summary_metrics.json').write_text(dumps(summary,indent=2))
    files=['forecasting_metrics.csv','sitewise_metrics.csv','model_to_operations_comparison.csv',
        'reserve_schedule_comparison.csv','robustness_results.csv','scaling_results.csv',
        'summary_metrics.json','optimization_summary.json']
    missing=[f for f in files if not (out/f).exists()]
    if missing:raise RuntimeError(f'Wait for completed legacy re-fit: {missing}')
    code_files=['run_mentor_validation.py','verify_legacy_evidence.py','finalize_mentor_evidence.py',
        'src/data/nasa_power.py','src/data/virtual_sites.py','src/reserve/simulator.py',
        'src/reliability/calibration.py','test_mentor_validation.py','config.yaml']
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*(out/f for f in files),*(Path(f) for f in code_files)]}
    manifest={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'scipy':scipy.__version__,'torch':torch.__version__,'sha256':hashes,
        'verification':'15 model fits, 1500 site metrics, 30 paired reserve runs, hourly constraints/accounting verified',
        'commands':['python run_mentor_validation.py','python verify_legacy_evidence.py',
                    'python -m unittest test_mentor_validation -v','python audit.py',
                    'python finalize_mentor_evidence.py']}
    (out/'reproducibility_manifest.json').write_text(dumps(manifest,indent=2))
    print('PASS: required evidence files, paired-model metrics, schedule constraints and energy accounting.')


if __name__=='__main__':main()
