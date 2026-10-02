# GridMesh: what the prototype actually does

## Big picture

The prototype answers two connected questions: how much solar might be available in 30 minutes,
and how limited backup should cover the expected demand gap plus forecast uncertainty. An operator
gets recommendations and warnings. No command goes to a real battery, generator or electricity grid.

## 1. Weather becomes a repeatable multi-site experiment

`src/data/adapter.py` loads the supplied ten-minute weather/irradiance record. The PV model converts
irradiance and temperature into modelled generation. `src/data/virtual_sites.py` varies site
capacity, configuration, sensor quality and history length. Fault scenarios corrupt copies,
preserving clean data for offline evaluation. This is simulated heterogeneity over shared weather.

`src/data/preprocessing.py` builds a one-hour feature history for a 30-minute target. Each month's
days form separate train/validation/test segments. Windows cannot cross those boundaries. Scaling
uses training rows only. A separate chronological split tests seasonal shift. The blocked-monthly
benchmark is not a claim of prospective year-long deployment performance.

## 2. Models correct a physical baseline

Smart persistence extrapolates the current clear-sky relationship. Learned models predict its
error: `future actual PV - baseline forecast`. Inference adds that correction to the baseline and
clips valid per-unit output to [0, 1]. For example, baseline 0.40 plus correction -0.08 gives 0.32.

The shallow tree tests a simple interpretable rule set. XGBoost adds many small trees sequentially,
each reducing remaining training error. Depth limits, regularization and validation early stopping
control complexity. Local XGBoost fits each site's rows separately. Pooled XGBoost and pooled MLP
are comparison references that require centralized raw data. They are not the proposed deployment.

MLP, GRU and LSTM provide neural alternatives. The common experiment reports all 12 methods under
the same final metrics. Validation ranks candidates; test reserve results never choose the model.
Stopping objectives and training budgets differ and are disclosed. Gain importance describes
predictive contributions, not causality.

## 3. Sites collaborate through compatible neural parameters

`src/federated/client.py` trains locally and reports parameter arrays plus scalar health/validation
information. `client_selection.py` limits round participation and can prioritize drift. Stable
sites may send small heartbeats rather than train. Dropout removes communication for that round.

`src/federated/server.py` rejects invalid/extreme updates and requires a positive-weight quorum.
`reliability_aware.py` weights accepted clients by sample count, sensor quality and evolving trust.
Quarantined clients contribute no ordinary update, but periodic probes permit recovery. Candidate
validation checks error, bias, worst-site performance and deterioration. Failed candidates retain
the trusted checkpoint. Clean simulation scores are reporting-only.

`hierarchical.py` composes neighbourhood and feeder weights. This single-process algebra matches
two-stage averaging. It can reduce traffic crossing the feeder boundary, while increasing total
traffic by adding a hop. It does not prove independent service fault isolation. XGBoost trees are
never parameter-averaged with FedAvg.

## 4. Invalid forecasts degrade explicitly

`src/reliability/safety.py` selects valid current, last-trusted, local or persistence output from
the candidates supplied by its caller. It records source, reason, age and operator attention.
Expired/stale/drift inputs force persistence and an alert. If the baseline is also invalid, it
raises an explicit error. Training rollback and final forecast fallback connect to the server.
The runtime helper exercises the complete ordering in tests; live local-checkpoint delivery and
gateway freshness integration remain future work. This is not a production model registry.

## 5. Forecast uncertainty becomes feasible reserve advice

`src/reserve/simulator.py` sums site forecasts and actuals with their capacities. Working with
aggregate residuals preserves the shared-weather correlation. For error `e=actual-forecast`,
the research-inspired Gaussian margin is `max(0, z_(1-delta)*sigma - mu)`. At delta .05,
mu -2 kW and sigma 6 kW, the illustrative margin is 11.87 kW.

Only residuals observable at forecast issue time enter rolling statistics. An empirical tail
margin avoids assuming Gaussian shape. `src/reliability/calibration.py` monitors previously
issued margins and applies a fixed floor when history/coverage deteriorates. No universal
time-series coverage guarantee follows.

The expected gap is `max(demand - planned grid import - forecast solar, 0)`. Required backup
adds the margin. A daily linear program allocates backup under MW power and MWh energy limits,
penalizing planned uncovered requirements. Evaluation applies scheduled backup against actual
solar and reports unserved energy, availability, grid imports, cost and capacity gaps.

This daily allocation uses a day's sequence of rolling forecasts retrospectively. Its calibration
is causal, but allocation is not yet a live day-ahead planner. A deployable pilot needs rolling
optimization with remaining energy, or genuine forecasts issued before the day begins. A wider
margin may exhaust scarce energy elsewhere, so guarded results stay separate from nominal results.

## 6. The dashboard explains the evidence

`build_dashboard.py` creates four historical replays. `dashboard.py` displays forecast, backup,
trust, dropout, rollback, calibration health and uncovered capacity, plus experiment evidence tabs.
`check_dashboard.py` checks all four scenarios. Training is precomputed before a demo.

## Reproduction

From the repository in PowerShell, use the README setup first, then:

```powershell
.\.venv\Scripts\python.exe audit.py
.\.venv\Scripts\python.exe run_tree_comparison.py --quick
.\.venv\Scripts\python.exe run_tree_comparison.py
.\.venv\Scripts\python.exe run_common_comparison.py --quick
.\.venv\Scripts\python.exe run_common_comparison.py
.\.venv\Scripts\python.exe run_reliability_stress.py --quick
.\.venv\Scripts\python.exe run_reliability_stress.py
.\.venv\Scripts\python.exe run_reliability_stress.py --scenarios sensor_recovery
.\.venv\Scripts\python.exe run_scaling.py --max-sites 500
.\.venv\Scripts\python.exe run_scale_stress.py
.\.venv\Scripts\python.exe run_operational_stress.py
.\.venv\Scripts\python.exe build_evidence_report.py
.\.venv\Scripts\python.exe run_reserve_benchmark.py --common
.\.venv\Scripts\python.exe build_dashboard.py
.\.venv\Scripts\python.exe check_dashboard.py
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Full matrices can take an hour or longer depending on hardware and concurrent work. Quick outputs
use distinct filenames. Do not run legacy reporting after the enhanced report without regenerating it.

See `MODEL_RELIABILITY_SCALABILITY.md` for experimental limits, `HACKATHON_ALIGNMENT.md` for the
organizer's criteria, `PITCH_NUMBERS.md` for current values, and `outputs/RESULTS.md` for failures
as well as successes. The strongest next step is measured Indian multi-site PV/load plus a shadow
pilot, not extra model families or production microservices.
