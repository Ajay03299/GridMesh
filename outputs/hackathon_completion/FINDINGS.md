# GridMesh implemented improvements and evidence



Frozen primary experiment reproduced: 70 test rows, 270 validation rows. Five-seed chronological daytime means: fixed20 96.89% supplied, 0.875 MWh ENS, 27.04 MWh scheduled backup; nominal 99.39%, 0.128 MWh, 16.76 MWh. Existing source/config/results unchanged.

Legacy 4-site/+30-minute, blocked-month 100-site reference, chronological 100-site/+60-minute and new operating stresses are separate protocols. No combined headline.



## Contribution and multi-seed fault comparison

Healthy-data primary RMSE (% of capacity): persistence 9.84, smart persistence 6.97, local-only 7.58, basic FedAvg 6.22, weighted FedAvg 6.23. Training budgets differ: this is not an equal-compute architecture ranking. Federation rationale is separate site governance, not proof of formal privacy.

Selective updates: 45.55% fewer modeled protocol bytes, +2.72% relative mean RMSE versus basic FedAvg. These are accounting estimates, not measured traffic.

New faults: three paired seeds 42–44, 100 virtual sites, 12 rounds, cap20. Twenty stale sites or 50% dropout. Freeze uses original validation-selected policy per method; no fault-specific retuning.

| scenario | method | healthy_rmse | availability_pct | ens_mwh | backup_mwh | cost_score |
|---|---|---|---|---|---|---|
| dropout_50pct | fedavg | 6.6541 | 99.523 | 0.0748 | 15.8357 | 17.3315 |
| dropout_50pct | reliability_fedavg | 6.6823 | 99.523 | 0.0751 | 15.6267 | 17.1278 |
| dropout_50pct | reliability_fedavg_event | 6.4928 | 99.4633 | 0.081 | 15.252 | 16.872 |
| stale_20_sites | fedavg | 6.3219 | 98.9267 | 0.2971 | 19.6665 | 25.6091 |
| stale_20_sites | reliability_fedavg | 6.7038 | 99.5826 | 0.0749 | 15.4889 | 16.987 |
| stale_20_sites | reliability_fedavg_event | 6.231 | 99.523 | 0.0678 | 15.5727 | 16.9296 |

Weighting does not establish a consistent improvement. It remains an experimental option; do not use the legacy 8.9% gain as current-India evidence.



## Paired operating sensitivity

Eight scenarios, five seeds 42–46. Demand errors are a 5% bias with 10% Gaussian standard deviation; grid loses 50% every sixth daytime interval; backup is unavailable every eighth daytime interval; known power is reduced30%; daily energy reduced50%; every sixth row has a stale forecast; solar loses35% every sixth daytime interval. Engineering assumptions, not empirically fitted field distributions.

Candidate margins and target preregistered. Joint-error quantile uses only prior observable demand, grid and solar errors. Select on validation: >=99% supply in EACH seed and ENS no worse than fixed20, then smallest scheduled backup. Freeze before test. No qualifying joint candidate means fixed20 fallback, not a guarantee. Simple comparator is validation-selected fixed10/15/20 using original no-worse reliability rule.

New joint variant combines net-error calibration AND stale-input safety bounds; it does not isolate their separate causal contributions. Original nominal default and all source forecasts/models unchanged.

v1 is an incomplete parser-error run. v2 used periodic calendar-hour events that could hit nights. v3 corrects events to predefined daytime ranks (not chosen by observed errors), making delivery faults relevant to the evaluated intervals. v2 is superseded, retained for audit, and excluded from final claims.



| scenario | role | availability_pct | ens_mwh | backup_mwh | cost_score |
|---|---|---|---|---|---|
| backup_derated | existing_nominal | 99.3918 | 0.1276 | 16.7359 | 19.2874 |
| backup_derated | fixed | 96.8873 | 0.8748 | 27.0319 | 44.5283 |
| backup_derated | validation_guard | 99.678 | 0.0524 | 11.3134 | 12.3618 |
| backup_derated | validation_simple | 98.712 | 0.0569 | 24.8289 | 25.9672 |
| backup_unavailable | existing_nominal | 98.9267 | 0.2373 | 16.7601 | 21.5068 |
| backup_unavailable | fixed | 96.78 | 0.8836 | 27.0358 | 44.7078 |
| backup_unavailable | validation_guard | 96.78 | 0.8836 | 27.0358 | 44.7078 |
| backup_unavailable | validation_simple | 98.3184 | 0.1659 | 24.8289 | 28.1478 |
| cloud_burst | existing_nominal | 97.8533 | 0.5598 | 25.2952 | 36.491 |
| cloud_burst | fixed | 96.8873 | 0.8781 | 27.0358 | 44.5985 |
| cloud_burst | validation_guard | 96.8873 | 0.8781 | 27.0358 | 44.5985 |
| cloud_burst | validation_simple | 98.712 | 0.058 | 24.8289 | 25.9879 |
| demand_error | existing_nominal | 93.5599 | 1.2756 | 16.7601 | 42.2715 |
| demand_error | fixed | 91.0197 | 2.4945 | 27.0358 | 76.9266 |
| demand_error | validation_guard | 91.0197 | 2.4945 | 27.0358 | 76.9266 |
| demand_error | validation_simple | 95.5277 | 0.9337 | 24.8289 | 43.5029 |
| grid_uncertainty | existing_nominal | 88.8014 | 5.9154 | 16.7601 | 135.0682 |
| grid_uncertainty | fixed | 87.7996 | 5.8572 | 27.0358 | 144.1793 |
| grid_uncertainty | validation_guard | 87.7996 | 5.8572 | 27.0358 | 144.1793 |
| grid_uncertainty | validation_simple | 88.8372 | 5.1326 | 24.8289 | 127.4806 |
| half_daily_energy | existing_nominal | 97.8891 | 0.5098 | 12.1776 | 22.3745 |
| half_daily_energy | fixed | 96.8873 | 0.8748 | 13.5443 | 31.0407 |
| half_daily_energy | validation_guard | 96.8873 | 0.8748 | 13.5443 | 31.0407 |
| half_daily_energy | validation_simple | 96.8873 | 0.8748 | 13.5179 | 31.0143 |
| normal | existing_nominal | 99.3918 | 0.1276 | 16.7601 | 19.3116 |
| normal | fixed | 96.8873 | 0.8748 | 27.0358 | 44.5322 |
| normal | validation_guard | 99.678 | 0.0524 | 11.361 | 12.4094 |
| normal | validation_simple | 98.712 | 0.0569 | 24.8289 | 25.9672 |
| stale_forecast | existing_nominal | 97.424 | 0.4886 | 18.0281 | 27.7995 |
| stale_forecast | fixed | 96.8873 | 0.8748 | 27.0358 | 44.5322 |
| stale_forecast | validation_guard | 99.9284 | 0.0115 | 10.3472 | 10.578 |
| stale_forecast | validation_simple | 98.6047 | 0.0621 | 24.8049 | 26.0461 |



Normal joint guard: 99.68% supplied, 0.052 MWh ENS, 11.36 MWh scheduled backup. Existing nominal: 99.39%, 0.128, 16.76. These are separate new sensitivity results, not replacements for the frozen primary benchmark.

Guard fallback in demand/grid/delivery/energy/cloud stresses shows validation target failure. Under grid loss, existing nominal improves supplied intervals over fixed but has MORE mean ENS (5.915 vs5.857 MWh). A single reliability metric can conceal degradation. Half energy increases nominal ENS to0.510 MWh. Unknown delivery failure cannot be fixed merely by rescheduling forecasts.

Test intervals: daylight only. No night adequacy, field outage hours, emissions, household savings or safety guarantee. Values are descriptive seed means, not confidence intervals. Failure counts overlap: margin undercoverage, constrained assets and exhausted allowance can coexist.



## Actual implementation

Missing, invalid, stale or degraded solar inputs produce an explicit zero-solar PLANNING BOUND with mandatory review; never display that bound as a valid forecast. Missing demand/assets still fail clearly. Single-interval allocation remains q=min(requirement,power,remaining energy/dt). Planned u=max(requirement-q,0). Ex-post ENS=max(actual demand-actual grid-actual solar-actually available scheduled backup,0)*dt.

Five deterministic 100-home scenarios now reuse the dashboard: normal, cloud, insufficient backup, missing/stale solar and illustrative degraded-site/dropout status. Confirm/reject is a proposed human action; no equipment command or simulated acknowledgement database.

Full-service YAML has payer, tax/source and upfront/monthly categories. Replacement and financing are alternative capital-recovery modes. No asset quote is invented; missing full cost/price/break-even remains TBD.



## Verification

22 regression unit tests passed. Broader audit36/36 passed. Operator-view15 scenario/interval paths passed. Frozen operational replay70/270 rows matched. Dashboard dark/light plus advisory smoke checks are recorded separately after final run.



## External validation still needed

Measured synchronized solar/load/grid/backup series and latency, owner permissions, technology-specific safety/replenishment, local operator/DISCOM review, installed supplier/tariff/tax quotes, funding agreement and willingness-to-pay interviews. No deployments, partnerships or subsidies exist in this evidence.