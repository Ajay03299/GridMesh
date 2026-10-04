"""Build researched itemised costs. Historical allocations are not pricing."""
import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
import yaml
from src.evaluation.pilot_costs import affordability_plan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/affordability_costs.yaml')
    parser.add_argument('--output',default='outputs/affordability_v1')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rows=[affordability_plan(cfg,b,a,h) for a in ('existing_assets','new_equipment')
          for b in ('low','base','high') for h in cfg['participation']]
    (out/'scenarios.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'sources_and_assumptions.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    columns=['band','asset_scenario','paying_homes','known_upfront_incremental_inr',
             'known_monthly_incremental_cash_inr','setup_recovery_monthly_inr',
             'contingency_monthly_inr','covered_service_monthly_inr',
             'covered_cost_recovery_per_home_inr','full_incremental_contribution_inr',
             'total_operating_monthly_inr','incremental_complete','total_complete']
    with (out/'scenarios.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerows({k:r[k] for k in columns} for r in rows)
    lines=['# GridMesh itemised affordability model','',
        'Editable input: configs/affordability_costs.yaml. Rerun build_full_service_costs.py.',
        'Planning estimates, not installed quotes, validated prices or proof of affordability.',
        'Existing-assets scope: paid incremental staff/support, added community connection, readable owner meters/exports. Retrofit and access remain unquoted.','',
        '| Scope | Band | Paying homes | Known setup INR | Covered service/month INR | Covered contribution/home INR | Complete incremental fee | Total operations |',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['asset_scenario']} | {r['band']} | {r['paying_homes']} | {r['known_upfront_incremental_inr']:.2f} | {r['covered_service_monthly_inr']:.2f} | {r['covered_cost_recovery_per_home_inr']:.2f} | {'TBD' if not r['incremental_complete'] else format(r['full_incremental_contribution_inr'],'.2f')} | {'TBD' if not r['total_complete'] else format(r['total_operating_monthly_inr'],'.2f')} |")
    lines+=['','## Base itemisation (existing assets)','',
        '| Item | Period | Quantity | Rate | Currency | Tax | Tax-inclusive INR | Monthly setup provision INR | Input status |',
        '|---|---|---:|---:|---|---:|---:|---:|---|']
    base=affordability_plan(cfg)
    for d in base['details']:
        value=lambda k:'TBD' if d[k] is None else f"{d[k]:.2f}"
        status=('Task/rate/tax assumptions' if d['source']=='task_estimate' else 'Quote needed' if d['source']=='quote_needed' else 'Published rate; eligibility/quantity assumptions')
        lines.append(f"| {d['name']} | {d['period']} | {value('quantity')} | {value('rate')} | {d['currency']} | {value('tax_fraction')} | {value('gross_inr')} | {value('recovery_monthly_inr')} | {status} |")
    lines+=['','## Participation and major-driver sensitivity','',
        '| Change to base estimate | Monthly covered cost INR | At 100 homes | At 50 homes |',
        '|---|---:|---:|---:|']
    changes=[('Base',None,None),('Operator review +15 paid hours/month','operator_review',15),
             ('Software support +4 paid hours/month','software_support',4),
             ('Known setup recovered over 12 months instead of 36','months',12)]
    sensitivity=[]
    for label,name,value in changes:
        variant=deepcopy(cfg)
        if name=='months':variant['setup_recovery_months']=value
        elif name:
            target=next(i for i in variant['items'] if i['name']==name)
            target['quantity']['base']+=value
        cost=affordability_plan(variant)['covered_service_monthly_inr']
        sensitivity.append(dict(change=label,monthly_inr=cost,homes100=cost/100,homes50=cost/50))
        lines.append(f'| {label} | {cost:.2f} | {cost/100:.2f} | {cost/50:.2f} |')
    fee_at_100=base['covered_service_monthly_inr']/100
    deficit=base['covered_service_monthly_inr']-50*fee_at_100
    conditional=deepcopy(cfg)
    conditional['funding'].update(sponsor_monthly_inr=deficit,
        sponsor_name='Hypothetical association/owner pilot fund, NOT agreed',
        sponsor_confirmed=False,proposed_household_payment_inr=fee_at_100)
    ordinary=affordability_plan(conditional,paying_homes=50)
    funded=affordability_plan(conditional,paying_homes=50,conditional_funding=True)
    (out/'conditional_funding.json').write_text(json.dumps(dict(
        scenario='50 payers at exact 100-payer base contribution',
        funding_agreed=False,unfunded=ordinary,conditional_only=funded),indent=2),encoding='utf-8')
    lines+=['',f'If only 50 pay the 100-home base contribution of INR{fee_at_100:.2f}, the covered-cost deficit is INR{deficit:.2f}/month, BEFORE unknown costs. Association reserves or a named sponsor must fund it; neither is confirmed.',
        'Each extra tax-inclusive INR1,000/month adds INR10/home at 100 payers or INR20/home at 50. No subsidy or DISCOM payment assumed.','',
        f'Conditional case only: a named hypothetical association fund contributes INR{deficit:.2f}/month, allowing50 homes to pay the100-home base contribution for covered costs. No agreement exists; ordinary outputs ignore unconfirmed funding. Full economics remain TBD.','',
        '## Interpretation','',
        'Covered service = paid recurring digital/people costs + known setup / recovery months + explicit contingency. Initial setup cash is separate; the provision is not a second setup cash payment.',
        'Setup contingency is funded through the recovery provision, not an already-funded upfront balance. Deposits/advance billing need a cash agreement.',
        'Full incremental cost additionally needs extra owner access, metering/edge retrofit recovery, and changes in consumed backup energy/asset wear. These remain TBD.',
        'Total operations additionally includes existing owner fees, physical O&M and consumed backup energy. Existing equipment capex is not recharged.',
        'New equipment needs an installed quote. Select capital recovery OR financing. Financing includes principal and interest, not another depreciation charge.',
        'Cost recovery has zero commercial margin, not a retail-fee recommendation or tax ruling. Household collection tax treatment is unconfirmed. No input-tax credits or free labour assumed.',
        'No energy/bill savings monetised: scheduled backup is not consumption. Simulated delivered backup is idealised dispatch, not invoice data. Abstract cost score never becomes INR.',
        'Conclusion: recovery of covered costs is calculable; complete economic feasibility and willingness to pay are unresolved. Start with an existing operator and measured data, not new equipment or an unsupported household promise.','',
        '## Sources and task assumptions','']
    for name,s in cfg['sources'].items():
        lines += [f'### {name}',f"Checked {s['checked']}; {s['location']}; {s['units']}.",
                  s['evidence'],s['eligibility'],str(s['url'] or 'Engineering assumption / quote needed, no market price claimed'),'']
    lines+=['## Quote / interview checklist','',
        '- Owner: current access/O&M contract, additional availability fee, responsibility for energy.',
        '- Technician: meter APIs, local training compute, gateway inventory, installed safety quotes and useful life.',
        '- Connectivity: exact address, permitted community use, tax-inclusive bill, deposits and commissioning.',
        '- Operator/support: paid hours, employment overhead, response scope and contractor tax status.',
        '- Energy: technology, measured dispatch/replenishment, charging efficiency/fuel and eligible tariff.',
        '- Association: upfront cash funder, collections, participation risk and shortfall contract.',
        '- Households: reliability needs, willingness to pay and income burden at 50/100 participation.',
        '- Accountant: invoice/collection tax treatment and capital recovery method.',
        'No interviews, enquiries, purchases or deployment undertaken.']
    (out/'COST_MODEL.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'sensitivity.json').write_text(json.dumps(sensitivity,indent=2),encoding='utf-8')
    print(json.dumps({k:base[k] for k in ['known_upfront_incremental_inr','covered_service_monthly_inr','covered_cost_recovery_per_home_inr','incremental_complete','total_complete']},indent=2))


if __name__=='__main__':main()
