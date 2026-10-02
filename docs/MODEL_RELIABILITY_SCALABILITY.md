# Model selection, reliability and scaling

## Architecture decision

Keep the professor's division: approximately 75% forecasting/federated learning and 25% one-node
reserve evaluation. Add strong tabular baselines before making the federated architecture more
complex. XGBoost is a baseline and a possible local fallback candidate. The collaborative protocol
uses neural models because FedAvg averages compatible parameter arrays. Averaging tree split
structures would be incorrect. Federated XGBoost remains a future tree-bagging/cyclic experiment.

The main evidence commands are `run_common_comparison.py`, `run_tree_comparison.py`,
`run_reliability_stress.py`, `run_scaling.py`, `run_scale_stress.py` and
`run_operational_stress.py`. `build_evidence_report.py` refuses to publish incomplete five-seed
model/reliability evidence. Quick modes write separate filenames.

## Dataset and fair comparison

The repository contains one 52,560-row NSRDB-format weather/irradiance file at 10-minute intervals.
The file does not establish its year or geographic location. Year 2019 is an explicit assumption.
Do not call this a verified Indian dataset. Obtain Indian measured PV/load data for the pilot.

PV uses a simplified horizontal-plane irradiance/temperature model, rather than measured
generation. The four benchmark sites vary sensor quality, meter noise, PV configuration and
training-history length. They all share the same weather. Five seeds vary these simulated
parameters and training randomness, not the weather locations.
Fault comparisons use clean test features/targets to isolate damage from corrupted training
and client-validation histories. Runtime stale/invalid inputs have separate direct guard tests.
These experiments do not constitute an end-to-end live faulty-sensor replay.

Each forecast looks 30 minutes ahead using one hour of lagged PV/weather. Only astronomical,
calendar and clear-sky covariates can come from the target timestamp. The blocked-monthly split
uses the first 70% of days in each month for training, the next 15% for validation and the remainder
for testing. Windows cannot cross segment boundaries. Each standardizer fits local training rows.
The chronological stress split tests a later season. No random time-series split is used.

The learned target is `actual_future_pu - smart_persistence_pu`. Inference adds the predicted
correction back to smart persistence and clips finite values to [0,1]. Non-finite values trigger
an explicit error or the forecast fallback layer. All final metrics use identical daytime test
timestamps and are expressed in per-unit installed capacity.

| Method | Training arrangement | Selection / purpose |
| --- | --- | --- |
| Persistence / smart persistence | No fitting | Physical/history baselines |
| Decision tree | Pooled rows | Depth 8, minimum leaf 30, transparent pooled baseline |
| Local XGBoost | One model per site | Local validation early stopping, no collaboration |
| Centralized XGBoost | Pooled rows | Pooled accuracy oracle |
| Local MLP | One model per site | Local validation checkpoint |
| Centralized MLP | Pooled rows | Pooled accuracy oracle |
| FedAvg / reliability-aware / event-aware MLP | Parameter exchange | Shared validation checkpoint |
| GRU / LSTM | Same reliability-aware FL protocol | Recurrent alternatives |

XGBoost uses histogram trees, learning rate .03, depth 6, min-child-weight 5, 80% row/feature
sampling, L2 penalty 1, configurable L1 penalty, at most 600 estimators and 40 validation rounds
without improvement. Quick mode caps at 250 estimators and 25 early-stopping rounds. CPU execution
and fixed seeds are supported. Local and pooled training call the same fitting/prediction helpers.

Final metrics are common, but training procedures have distinct budgets: 15 epochs for local/pooled
neural models, 20 FL rounds with one local epoch, and validation-stopped boosting. XGBoost early
stopping monitors validation residual RMSE on all validation rows, while neural checkpoints monitor
daytime reconstructed forecasts. That distinction is disclosed rather than claiming identical
optimization. Mean validation RMSE ranks candidates; reserve outcomes do not choose a model from
test data. Results are conditional on this fixed configuration, not a comprehensive hyperparameter
search. Feature importances use XGBoost gain and decision-tree impurity decrease. Grouping lagged
variables helps readability, but correlated features and gain do not establish causal effects.

## Runtime reliability layers

Data quality identifies missing/frozen readings, and the existing fault copies model irradiance
corruption, noisy meters, stale sensors and calibration bias. Update screening checks shape,
finite values, positive sample count, finite validation scores and a norm threshold relative to
the peer median. This is an availability heuristic. It does not establish Byzantine security,
detect every malicious update or protect against collusion.

Weights combine training sample count, an EMA of peer-relative validation trust and sensor quality.
Low-trust clients contribute zero weight. They skip ordinary rounds and can send recovery probes
every three rounds. Healthy reports restore trust. An all-quarantined round retains the current
model. A minimum of two positive-weight updates is required.

The server asks contributing clients to score a candidate on their available validation data.
It checks finite predictions, per-site RMSE, bias, a smart-persistence comparison and degradation
relative to the prior trusted score. Clean simulation validation is reporting-only and never
controls acceptance. Failed candidates retain the trusted checkpoint and log the reason. These
fixed thresholds are permissive guardrails, not a claim of optimal model selection.

The runtime forecast helper chooses current trusted output, last-trusted output, local output,
then smart persistence from the supplied available candidates. It records source, reason, age
and operator attention. Expired forecasts, stale sensors or a drift alarm use persistence and
an alert. If even persistence is invalid, it fails explicitly. The server connects this helper
to final forecast generation, and direct stress tests exercise stale/expired/invalid cases.
Local checkpoint transport and live freshness checks remain integration work: this simulation
does not pretend to have live gateway timestamps or a production model registry.

Reserve monitoring evaluates previously issued margins only after their outcome becomes
observable. It reports rolling coverage, sample count, calibration age and warnings. A predefined
fixed-margin floor responds to insufficient/stale history or poor coverage. The guarded policy
is separate from nominal n-sigma so users can assess its cost and scarce-energy consequences.
Neither rolling Gaussian nor empirical calibration gives universal finite-sample coverage for
autocorrelated errors.

## Hierarchical simulation

Logical sites form groups of 25. Each neighbourhood applies its own trust/quality weights, then
the feeder combines group models using represented sample volume, group quality and group trust.
The composed site weights are mathematically equivalent to the two successive parameter averages.
An audit checks this equivalence. Empty or zero-trust groups contribute no weight. Site checks
protect intermediate arrays from non-finite updates. Group health and weights are returned by
the aggregator, and feeder traffic is logged separately from local traffic.

This is a synchronous single-process algebraic simulation. It does not run independent network
services, add asynchronous queues or provide geographic failure isolation. The unavailable-group
test removes its clients from communication and retains them in forecast evaluation, with stale
output falling back to persistence. More realistic group outages and recovery timing are pilot
experiments. No superior accuracy claim follows from hierarchical aggregation.

## Scale protocol and measurement limits

The scale matrix uses 4, 20, 50, 100, 250 and 500 clients with three rounds and seed 42. All methods
have a cap of 20 selected clients before event skips/dropout. Drift clients get priority within
that cap. When more drift clients request service than the cap allows, the selection samples them
rather than breaking the resource limit. There is no measured site-type diversity guarantee.

At 4/20 clients the data are full virtual-site datasets. At larger sizes, four compact reference
datasets are shared under new identities. Each template retains 2,000 training rows and 1,000
validation/test rows using deterministic time subsampling. This is a communication/runtime stress
test. Its accuracy is not comparable to the full-resolution four-site test or to 500 real sites.

Traffic includes current-model download, local-update upload, candidate-validation download,
64 bytes of scalar metadata and 16-byte status heartbeats. Hierarchical totals include both local
and feeder hops. Protocol framing, TLS, retries and network latency are excluded. Traffic is an
accounted payload estimate, not captured packet traffic. Runtime separates client fitting,
coordinator/bookkeeping and final inference. Memory is process peak working set, cumulative over
the run. Concurrent experiments affect timings. Three rounds do not establish long-run convergence.

`run_scale_stress.py` adds 25/50% dropout, 10/20% stale sites, combined failures and a fully
unavailable first neighbourhood at 100 logical sites. `run_operational_stress.py` holds a learned
forecast fixed and stresses common cloud loss, imports and backup energy. Results and failed targets
remain in the generated evidence report. A 1,000-site run is not part of the published evidence.

## Affordability and deployment

The existing illustrative software service budget is INR4,650/month, or INR46.50 per household
at 100 paying households. At 50 households it is INR93. These figures are hypotheses, not supplier
quotes or proven willingness to pay. PV, gateways, installation, reserve hardware, energy, grid
bills, tax and finance are outside that service budget. A community operator/ESCO would own daily
monitoring and support, with escalation to the DISCOM and approved asset operators.

The default experimental sites are plant-sized (5–20 MW), independent of the household service
cost allocation. A neighbourhood pilot must replace capacities with measured kW-scale resources,
replace demand/limits/costs and verify that benefits exceed the total service and hardware cost.
The simulator's linear power scaling does not establish economic or physical transferability.

References: [XGBoost API](https://xgboost.readthedocs.io/en/stable/python/python_api.html),
[FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html),
[reserve research](https://doi.org/10.1093/ce/zkag018),
[Flower tree-specific FL examples](https://flower.ai/docs/examples/xgboost-comprehensive.html).
