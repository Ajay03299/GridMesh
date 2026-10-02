# Submission alignment update

This working copy strengthens the original GridMesh prototype for the Yuva Yodha first-round pitch.

## Forecasting and federated learning

- Added configurable MLP, GRU and LSTM forecasters under one feature and federated protocol.
- Added `run_model_comparison.py` for a common-protocol architecture screen.
- Retained the compact MLP as the published default because it currently gives the best error and
  smallest communication payload in the quick screening run.
- Kept reliability-aware aggregation, event-aware participation, fault injection and the 100-site
  scale test as the core 75% of the technical story.

## Reserve optimization

- Corrected the interpretation of the research rule: `max(alpha*sigma - mu, 0)` is an uncertainty
  margin, not the complete reserve schedule.
- Added a causal empirical tail-quantile policy as a non-Gaussian comparison.
- Added the expected demand-supply gap before the uncertainty margin.
- Added explicit grid-import, backup-power and daily backup-energy limits.
- Added a daily linear program that schedules backup and exposes any planned capacity gap.
- Added configurable peak priority to remove arbitrary schedules when the daily energy limit binds.
- Added a five-seed reserve benchmark script.

## Evidence

- Five-seed n-sigma 5% result versus the fixed 20%-of-demand margin:
  - 34.9% less scheduled backup energy
  - 10.3% less unserved energy
  - 17.2% lower assumed total cost
- Added three-model acceptance checks and constrained-scheduler checks.
- Final audit: 24/24 checks pass.
- Updated dashboard data and labels so operators can see grid import, expected gap, uncertainty
  margin, scheduled backup and capacity warnings separately.

## Scope boundaries

- Weather and irradiance are real; PV output is modelled.
- Sites, faults, demand, capacities, asset limits and costs are simulated or assumed unless a real
  multi-site file is supplied.
- Raw histories remain local by design, but secure aggregation and differential privacy are not yet
  implemented.
- The reserve experiment covers daytime renewable-intermittency windows, not full feeder adequacy.
- GridMesh is positioned as a focused research module that could complement broader Schneider
  platforms after measured-data validation, security review and approved integration.
