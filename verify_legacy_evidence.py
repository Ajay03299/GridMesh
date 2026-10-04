"""Re-fit retained slide-7/9 protocols without overwriting historic experiment tables."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from run_simulation import run_method
from run_common_comparison import validation_rmse
from run_scaling import prepare_protocol_clients
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import evaluate,predictions_frame
from src.reserve.simulator import load_node,run_policies
from src.evaluation.serialization import dumps


def main():
    cfg=load_config();out=Path('outputs/mentor_validation');out.mkdir(parents=True,exist_ok=True)
    rows=[];ops=[]
    for seed in cfg['experiments']['seeds']:
        clients,_=prepare_clients(cfg,4,seed)
        # Fixed before looking at test outcomes; match retained 20-round legacy protocol.
        for method in ('reliability_fedavg','centralized_xgboost'):
            res=run_method(method,clients,cfg,seed,verbose=False)
            m=evaluate(res,clients)
            rows.append({'seed':seed,'scenario':'healthy','method':method,
                'global_rmse':m['global_rmse'],'healthy_rmse':m['healthy_rmse'],
                'validation_rmse':validation_rmse(res,clients)})
            if method=='reliability_fedavg':
                for _,s in run_policies(load_node(predictions_frame(res,clients)),cfg,seed):
                    if s['policy'] in ('fixed_20pct','nsigma_d0.05'):
                        ops.append({'seed':seed,**s})
        bad,_=prepare_clients(cfg,4,seed,[2,3],'stale',2)
        for method in ('fedavg','reliability_fedavg'):
            res=run_method(method,bad,cfg,seed,[2,3],verbose=False);m=evaluate(res,bad)
            rows.append({'seed':seed,'scenario':'stale_50pct','method':method,
                         'global_rmse':m['global_rmse'],'healthy_rmse':m['healthy_rmse']})
        drop=copy.deepcopy(cfg);drop['federated']['dropout_rate']=.5
        res=run_method('reliability_fedavg',clients,drop,seed,verbose=False);m=evaluate(res,clients)
        rows.append({'seed':seed,'scenario':'dropout_50pct','method':'reliability_fedavg',
                     'global_rmse':m['global_rmse'],'healthy_rmse':m['healthy_rmse']})
        pd.DataFrame(rows).to_csv(out/'robustness_results.csv',index=False)
        pd.DataFrame(ops).to_csv(out/'legacy_reserve_reproduction.csv',index=False)
        print(f'legacy verification complete seed={seed}',flush=True)
    # Same repeated-data scale protocol as retained 43.3% result. No geographic claim.
    sc=copy.deepcopy(cfg);sc['federated'].update(rounds=3,max_clients_per_round=20)
    clients=prepare_protocol_clients(sc,100,42);scale=[]
    for method in ('fedavg','reliability_fedavg_event'):
        res=run_method(method,clients,sc,42,verbose=False);m=evaluate(res,clients)
        scale.append({'n_sites':100,'seed':42,'method':method,'rounds':3,
            'global_rmse':m['global_rmse'],'comm_mb_per_round':m['mean_comm_kb_per_round']/1000,
            'clients_cap':20,'data_mode':'logical identities, four repeated compact histories',
            'sites_per_round':res.extra['rounds'].n_active.mean()})
    pd.DataFrame(scale).to_csv(out/'scaling_results.csv',index=False)
    r=pd.DataFrame(rows);o=pd.DataFrame(ops).groupby('policy').mean(numeric_only=True)
    healthy=r[(r.scenario=='healthy')&(r.method=='reliability_fedavg')].global_rmse.mean()
    stale=r[r.scenario=='stale_50pct'].groupby('method').healthy_rmse.mean()
    a,b=scale
    summary={'seeds':[42,43,44,45,46],'legacy_mlp_test_rmse_pct':100*healthy,
        'legacy_pooled_xgboost_validation_rmse_pct':100*r[(r.scenario=='healthy')&
            (r.method=='centralized_xgboost')].validation_rmse.mean(),
        'healthy_error_reduction_stale_pct':100*(1-stale['reliability_fedavg']/stale['fedavg']),
        'dropout_relative_error_increase_pct':100*(r[r.scenario=='dropout_50pct'].global_rmse.mean()/healthy-1),
        'scale_payload_reduction_pct':100*(1-b['comm_mb_per_round']/a['comm_mb_per_round']),
        'scale_relative_error_increase_pct':100*(b['global_rmse']/a['global_rmse']-1),
        'reserve':o.reset_index().to_dict('records'),
        'limitations':'Legacy 4-site unknown-location 10-minute weather. Scaling repeats data. Not Indian measurement validation.'}
    (out/'legacy_verification.json').write_text(dumps(summary,indent=2))
    print(dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
