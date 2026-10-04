# Evidence improvement results

## Frozen reference verified (unchanged)

Bengaluru 100 virtual sites, monthly blocked offline evaluation; retrospective full-day LP. Values below are per-seed daytime test means ± sample standard deviation. Availability is the percentage of fully supplied intervals, not measured uptime. Cost score uses assumed units, not INR.

| Comparison | Fully supplied % | Realized ENS MWh | Scheduled backup MWh | Cost score |
|---|---:|---:|---:|---:|
| fixed_20pct | 98.4644 ± 0.8312 | 0.2050 ± 0.1724 | 23.8048 ± 0.8819 | 27.9056 ± 3.6758 |
| nsigma_d0.05 | 97.6030 ± 0.5832 | 0.2689 ± 0.0644 | 18.2622 ± 1.7765 | 23.6405 ± 2.7935 |

### Why nominal reserve loses reliability

A tail-error margin is not a service guarantee. Small rolling windows and a Gaussian approximation can under-cover solar errors. Even when the requirement covers the actual deficit, power/daily-energy caps can prevent scheduling it. With constant uncovered-reserve penalties, the full-day LP can have multiple equally optimal time allocations: its objective does not directly maximize fully supplied intervals or minimize realized ENS. Neither actual solar nor a service-availability objective enters that planning LP. Wider margins alone therefore do not guarantee better service.

- fixed_20pct: mean 8.2 missed intervals/seed; 0.0 with too-small requirement; 8.2 despite adequate requirement (allocation/asset limits).
- nsigma_d0.05: mean 12.8 missed intervals/seed; 7.4 with too-small requirement; 5.4 despite adequate requirement (allocation/asset limits).

## Separate chronological experiment

Train first 70%, validation next 15%, test final 15% of the hourly record; windows cannot cross split boundaries. Training precedes test. Single-interval analytic LP uses current forecast, known synthetic demand/grid, earlier observed residuals and remaining daily scheduled-energy budget; no future-day forecast sequence. Demand is assumed perfectly forecast, not a measured load model. This changes planning/split protocol and must NOT be compared as a direct improvement to the frozen reference.

### Primary policy comparison: reliability_fedavg MLP, same causal planner

| Comparison | Fully supplied % | Realized ENS MWh | Scheduled backup MWh | Cost score |
|---|---:|---:|---:|---:|
| fixed_20pct | 96.8873 ± 0.2040 | 0.8748 ± 0.0339 | 27.0358 ± 1.0007 | 44.5322 ± 1.2293 |
| nominal_05 | 99.3918 ± 0.2400 | 0.1276 ± 0.0730 | 16.7601 ± 3.2420 | 19.3116 ± 4.4132 |

### Selected policy for each forecasting baseline

| Comparison | Fully supplied % | Realized ENS MWh | Scheduled backup MWh | Cost score |
|---|---:|---:|---:|---:|
| fedavg | 99.3918 ± 0.2400 | 0.1302 ± 0.0749 | 17.0342 ± 3.1928 | 19.6376 ± 4.4168 |
| local_only | 99.4633 ± 0.4195 | 0.1229 ± 0.1104 | 17.6564 ± 4.0094 | 20.1140 ± 5.8747 |
| persistence | 98.4258 ± 0.2939 | 0.0895 ± 0.0267 | 24.9205 ± 0.9228 | 26.7099 ± 0.5715 |
| reliability_fedavg | 99.3918 ± 0.2400 | 0.1276 ± 0.0730 | 16.7601 ± 3.2420 | 19.3116 ± 4.4132 |
| reliability_fedavg_event | 99.4275 ± 0.3878 | 0.1277 ± 0.0937 | 16.2421 ± 4.3934 | 18.7954 ± 6.1569 |
| smart_persistence | 98.0322 ± 0.4899 | 0.1816 ± 0.0405 | 25.2253 ± 0.9201 | 28.8573 ± 0.3487 |

### Paired policy test outcome (selected vs fixed, same chronological planner)

- fedavg: selected nominal_05; availability delta +2.504 percentage points; ENS delta -0.74465 MWh; backup delta -10.002 MWh; cost-score delta -24.895. No reliability degradation in every test seed: True. Lower mean backup: True.
- local_only: selected nominal_05; availability delta +2.576 percentage points; ENS delta -0.75194 MWh; backup delta -9.379 MWh; cost-score delta -24.418. No reliability degradation in every test seed: True. Lower mean backup: True.
- persistence: selected fixed_10pct; availability delta +1.538 percentage points; ENS delta -0.78535 MWh; backup delta -2.115 MWh; cost-score delta -17.822. No reliability degradation in every test seed: True. Lower mean backup: True.
- reliability_fedavg: selected nominal_05; availability delta +2.504 percentage points; ENS delta -0.74725 MWh; backup delta -10.276 MWh; cost-score delta -25.221. No reliability degradation in every test seed: True. Lower mean backup: True.
- reliability_fedavg_event: selected nominal_05; availability delta +2.540 percentage points; ENS delta -0.74716 MWh; backup delta -10.794 MWh; cost-score delta -25.737. No reliability degradation in every test seed: True. Lower mean backup: True.
- smart_persistence: selected fixed_10pct; availability delta +1.145 percentage points; ENS delta -0.69322 MWh; backup delta -1.810 MWh; cost-score delta -15.675. No reliability degradation in every test seed: True. Lower mean backup: True.

### Forecast baselines

Daytime test RMSE/MAE as % of installed site capacity. Local-only: 12 full epochs/site; FL: 12 rounds × up to 20 clients × 1 local epoch. These are documented budgets, not equal total compute. Basic FedAvg retains existing shared safety gates; it is a weighting baseline, not an entirely unguarded algorithm.

- fedavg: RMSE 6.221 ± 1.156%; MAE 4.586 ± 0.950%; mean modeled protocol traffic 6.009 MB/run.
- local_only: RMSE 7.575 ± 1.462%; MAE 5.560 ± 1.251%; mean modeled protocol traffic 0.000 MB/run.
- persistence: RMSE 9.843 ± 0.018%; MAE 8.666 ± 0.017%; mean modeled protocol traffic 0.000 MB/run.
- reliability_fedavg: RMSE 6.229 ± 1.157%; MAE 4.581 ± 0.966%; mean modeled protocol traffic 6.009 MB/run.
- reliability_fedavg_event: RMSE 6.391 ± 1.096%; MAE 4.740 ± 0.920%; mean modeled protocol traffic 3.272 MB/run.
- smart_persistence: RMSE 6.965 ± 0.014%; MAE 5.116 ± 0.010%; mean modeled protocol traffic 0.000 MB/run.

Selective FL vs basic FedAvg: 45.55% lower modeled protocol bytes/run with +2.72% relative RMSE change (ratios of five-seed means; not physical-network measurements).
The healthy-data reliability-weighted method is not more accurate than basic FedAvg on average. FL improves mean error versus smart persistence/local-only here, but smart persistence beats FL in seed 43. The selected persistence policy has lower mean ENS than selected FL despite lower fully-supplied availability and higher backup. No model is a universal operational winner.

### Same-rule model comparison (nominal_05 for every method)

| Comparison | Fully supplied % | Realized ENS MWh | Scheduled backup MWh | Cost score |
|---|---:|---:|---:|---:|
| fedavg | 99.3918 ± 0.2400 | 0.1302 ± 0.0749 | 17.0342 ± 3.1928 | 19.6376 ± 4.4168 |
| local_only | 99.4633 ± 0.4195 | 0.1229 ± 0.1104 | 17.6564 ± 4.0094 | 20.1140 ± 5.8747 |
| persistence | 96.9231 ± 0.1497 | 0.8725 ± 0.0341 | 26.5753 ± 0.9815 | 44.0247 ± 1.2473 |
| reliability_fedavg | 99.3918 ± 0.2400 | 0.1276 ± 0.0730 | 16.7601 ± 3.2420 | 19.3116 ± 4.4132 |
| reliability_fedavg_event | 99.4275 ± 0.3878 | 0.1277 ± 0.0937 | 16.2421 ± 4.3934 | 18.7954 ± 6.1569 |
| smart_persistence | 97.2093 ± 0.1600 | 0.8129 ± 0.0301 | 26.2573 ± 0.9618 | 42.5160 ± 1.1340 |

Healthy co-located data alone does not prove fault-handling benefit. Use the separately labeled legacy stress experiments for their original robustness claims. Do not generalize selective-update savings to real networks or geographic generalization.

### Bounded current-India failure smoke

Seed 42 only; 100 virtual sites; same chronological data. Twenty stale training copies or 50% random dropout; clean test targets. Policies remain frozen from the clean validation run; no fault/test re-tuning.

- dropout_50pct, fedavg: healthy RMSE 5.438%; fully supplied 99.284%; ENS 0.1106 MWh; backup 20.311 MWh; cost 22.524; traffic 2.729 MB.
- dropout_50pct, reliability_fedavg: healthy RMSE 5.406%; fully supplied 99.284%; ENS 0.1117 MWh; backup 20.249 MWh; cost 22.482; traffic 2.729 MB.
- dropout_50pct, reliability_fedavg_event: healthy RMSE 5.536%; fully supplied 99.284%; ENS 0.1622 MWh; backup 20.469 MWh; cost 23.713; traffic 1.694 MB.
- stale_20_sites, fedavg: healthy RMSE 5.413%; fully supplied 99.821%; ENS 0.0756 MWh; backup 17.812 MWh; cost 19.323; traffic 6.009 MB.
- stale_20_sites, reliability_fedavg: healthy RMSE 5.500%; fully supplied 99.284%; ENS 0.1284 MWh; backup 20.732 MWh; cost 23.301; traffic 6.009 MB.
- stale_20_sites, reliability_fedavg_event: healthy RMSE 5.441%; fully supplied 99.463%; ENS 0.0878 MWh; backup 19.738 MWh; cost 21.494; traffic 3.722 MB.

This one-seed smoke cannot establish statistically general robustness or service uptime.

## Evidence boundaries

Virtual sites, one gridded-weather record, modeled PV, synthetic demand, assumed available generic backup, no measured outage hours, no formal privacy, no geographic holdout. Policy selected solely on validation and frozen before operational test evaluation. Test metrics are descriptive; five seeds are not an independent-field confidence interval. All disappointing outcomes remain in the report.
