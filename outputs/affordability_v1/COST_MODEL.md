# GridMesh itemised affordability model

Editable input: configs/affordability_costs.yaml. Rerun build_full_service_costs.py.
Planning estimates, not installed quotes, validated prices or proof of affordability.
Existing-assets scope: paid incremental staff/support, added community connection, readable owner meters/exports. Retrofit and access remain unquoted.

| Scope | Band | Paying homes | Known setup INR | Covered service/month INR | Covered contribution/home INR | Complete incremental fee | Total operations |
|---|---|---:|---:|---:|---:|---|---|
| existing_assets | low | 50 | 7970.00 | 6996.23 | 139.92 | TBD | TBD |
| existing_assets | low | 100 | 7970.00 | 6996.23 | 69.96 | TBD | TBD |
| existing_assets | base | 50 | 21650.00 | 15268.45 | 305.37 | TBD | TBD |
| existing_assets | base | 100 | 21650.00 | 15268.45 | 152.68 | TBD | TBD |
| existing_assets | high | 50 | 55120.00 | 40874.32 | 817.49 | TBD | TBD |
| existing_assets | high | 100 | 55120.00 | 40874.32 | 408.74 | TBD | TBD |
| new_equipment | low | 50 | 7970.00 | 6996.23 | 139.92 | TBD | TBD |
| new_equipment | low | 100 | 7970.00 | 6996.23 | 69.96 | TBD | TBD |
| new_equipment | base | 50 | 21650.00 | 15268.45 | 305.37 | TBD | TBD |
| new_equipment | base | 100 | 21650.00 | 15268.45 | 152.68 | TBD | TBD |
| new_equipment | high | 50 | 55120.00 | 40874.32 | 817.49 | TBD | TBD |
| new_equipment | high | 100 | 55120.00 | 40874.32 | 408.74 | TBD | TBD |

## Base itemisation (existing assets)

| Item | Period | Quantity | Rate | Currency | Tax | Tax-inclusive INR | Monthly setup provision INR | Input status |
|---|---|---:|---:|---|---:|---:|---:|---|
| cloud_compute | monthly | 1.00 | 24.00 | USD | 0.18 | 2690.40 | 0.00 | Published rate; eligibility/quantity assumptions |
| offsite_storage | monthly | 1.00 | 3.00 | USD | 0.18 | 336.30 | 0.00 | Published rate; eligibility/quantity assumptions |
| community_connectivity | monthly | 1.00 | 499.00 | INR | 0.18 | 588.82 | 0.00 | Published rate; eligibility/quantity assumptions |
| operator_review | monthly | 15.00 | 250.00 | INR | 0.00 | 3750.00 | 0.00 | Task/rate/tax assumptions |
| software_support | monthly | 4.00 | 750.00 | INR | 0.18 | 3540.00 | 0.00 | Task/rate/tax assumptions |
| data_health_maintenance | monthly | 2.00 | 750.00 | INR | 0.18 | 1770.00 | 0.00 | Task/rate/tax assumptions |
| onboarding_integration | upfront | 20.00 | 750.00 | INR | 0.18 | 17700.00 | 491.67 | Task/rate/tax assumptions |
| operator_training | upfront | 4.00 | 250.00 | INR | 0.00 | 1000.00 | 27.78 | Task/rate/tax assumptions |
| connectivity_commissioning_allowance | upfront | 1.00 | 2500.00 | INR | 0.18 | 2950.00 | 81.94 | Task/rate/tax assumptions |
| incremental_meter_gateway_retrofit | upfront | 1.00 | TBD | INR | TBD | TBD | TBD | Quote needed |
| incremental_asset_access | monthly | 1.00 | TBD | INR | TBD | TBD | TBD | Quote needed |
| existing_asset_maintenance | monthly | 1.00 | TBD | INR | TBD | TBD | TBD | Quote needed |
| existing_asset_access | monthly | 1.00 | TBD | INR | TBD | TBD | TBD | Quote needed |
| consumed_backup_energy | usage | TBD | TBD | INR | TBD | TBD | TBD | Quote needed |

## Participation and major-driver sensitivity

| Change to base estimate | Monthly covered cost INR | At 100 homes | At 50 homes |
|---|---:|---:|---:|
| Base | 15268.45 | 152.68 | 305.37 |
| Operator review +15 paid hours/month | 19580.95 | 195.81 | 391.62 |
| Software support +4 paid hours/month | 19339.45 | 193.39 | 386.79 |
| Known setup recovered over 12 months instead of 36 | 16651.64 | 166.52 | 333.03 |

If only 50 pay the 100-home base contribution of INR152.68, the covered-cost deficit is INR7634.22/month, BEFORE unknown costs. Association reserves or a named sponsor must fund it; neither is confirmed.
Each extra tax-inclusive INR1,000/month adds INR10/home at 100 payers or INR20/home at 50. No subsidy or DISCOM payment assumed.

Conditional case only: a named hypothetical association fund contributes INR7634.22/month, allowing50 homes to pay the100-home base contribution for covered costs. No agreement exists; ordinary outputs ignore unconfirmed funding. Full economics remain TBD.

## Interpretation

Covered service = paid recurring digital/people costs + known setup / recovery months + explicit contingency. Initial setup cash is separate; the provision is not a second setup cash payment.
Setup contingency is funded through the recovery provision, not an already-funded upfront balance. Deposits/advance billing need a cash agreement.
Full incremental cost additionally needs extra owner access, metering/edge retrofit recovery, and changes in consumed backup energy/asset wear. These remain TBD.
Total operations additionally includes existing owner fees, physical O&M and consumed backup energy. Existing equipment capex is not recharged.
New equipment needs an installed quote. Select capital recovery OR financing. Financing includes principal and interest, not another depreciation charge.
Cost recovery has zero commercial margin, not a retail-fee recommendation or tax ruling. Household collection tax treatment is unconfirmed. No input-tax credits or free labour assumed.
No energy/bill savings monetised: scheduled backup is not consumption. Simulated delivered backup is idealised dispatch, not invoice data. Abstract cost score never becomes INR.
Conclusion: recovery of covered costs is calculable; complete economic feasibility and willingness to pay are unresolved. Start with an existing operator and measured data, not new equipment or an unsupported household promise.

## Sources and task assumptions

### aws
Checked 2026-10-04; Asia Pacific Mumbai; USD/month.
Public IPv4 Linux 4GB $24; 8GB $44. Object storage 5/100/250GB $1/$3/$5. Regular on-demand rates, no free trial credited.
Account and chosen region required. Mumbai transfer quota is half published headline allowance. No load/traffic benchmark on these SKUs. Local training compute must exist or be separately costed.
https://aws.amazon.com/lightsail/pricing/

### aws_tax
Checked 2026-10-04; India billing address; fraction of invoice.
India GST listed at 18%. No input-tax credit assumed.
Check entity / invoice at purchase. This is not a tax opinion for household contributions.
https://aws.amazon.com/tax-help/apac-vat-rates/

### airtel
Checked 2026-10-04; Bangalore; INR/month before GST.
Regular 499 plan, up to 40 Mbps; no promotion or OTT plan counted.
Address feasibility and permitted community use not confirmed. One community connection, NOT connectivity to 100 separate sites. No free installation credited.
https://www.airtel.in/plans/broadband/bangalore/

### telecom_tax
Checked 2026-10-04; India; fraction of invoice.
18% telecom GST; supplier also states GST additional.
Tax-inclusive cash cost. No input-tax credit assumed.
https://cbic-gst.gov.in/pdf/press-release/GST-telecom-services.pdf

### task_estimate
Checked 2026-10-04; Proposed Bengaluru pilot; hours x INR/hour.
Explicit engineering task estimates, not observed times, wage surveys, legal minimum wages or supplier quotes.
Negotiate paid incremental hours; do not assume existing staff work for free. Contractor 18% tax is a planning allowance, not established tax applicability.
Engineering assumption / quote needed, no market price claimed

### quote_needed
Checked 2026-10-04; Pilot address / asset owner TBD; item specific.
No installed quote, agreement, meter inventory or measured dispatch available.
Required to close total economics; null does not mean zero.
Engineering assumption / quote needed, no market price claimed

## Quote / interview checklist

- Owner: current access/O&M contract, additional availability fee, responsibility for energy.
- Technician: meter APIs, local training compute, gateway inventory, installed safety quotes and useful life.
- Connectivity: exact address, permitted community use, tax-inclusive bill, deposits and commissioning.
- Operator/support: paid hours, employment overhead, response scope and contractor tax status.
- Energy: technology, measured dispatch/replenishment, charging efficiency/fuel and eligible tariff.
- Association: upfront cash funder, collections, participation risk and shortfall contract.
- Households: reliability needs, willingness to pay and income burden at 50/100 participation.
- Accountant: invoice/collection tax treatment and capital recovery method.
No interviews, enquiries, purchases or deployment undertaken.