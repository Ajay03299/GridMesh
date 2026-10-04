# GridMesh — Neighbourhood Solar Reliability with Federated Learning

Hackathon prototype · Yuva Yodha Energy Tech Hackathon · Grid Reliability & Renewable Intermittency track.

**Substantive evidence improvements:** [reproduction, weakness checklist and pilot gaps](docs/EVIDENCE_IMPROVEMENTS.md).
Run `python run_evidence_improvement.py`, then `python report_evidence_improvement.py` for a separate
chronological Bengaluru baseline/policy experiment. Select **Deterministic advisory demo** in the
dashboard for an untrained cloud-shock/asset-limit scenario. Pilot economics are editable in
`configs/affordability_costs.yaml`; see [itemised affordability and historical correction](docs/AFFORDABILITY.md).
Run `python build_full_service_costs.py` for low/base/high,50/100-home scenarios.
The unsupported4650 allocation is withdrawn; full costs and willingness to pay remain unresolved.
Existing research/mentor evidence is unchanged.

**Current chronological extension:** [verified findings](outputs/hackathon_completion/FINDINGS.md).
**Separate historical reference evidence:** [`outputs/RESULTS.md`](outputs/RESULTS.md). Methodology and limits:
[`docs/MODEL_RELIABILITY_SCALABILITY.md`](docs/MODEL_RELIABILITY_SCALABILITY.md).

The enhanced branch adds decision trees, local/pooled XGBoost, a 12-method comparison, update
screening, rollback, forecast fallback, causal calibration monitoring and two-stage hierarchical
aggregation. The scale matrix reaches 500 logical clients; larger tests reuse reference datasets
and must not be presented as 500 independently measured sites.

## Windows setup and new evidence commands

From this repository folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe audit.py
.\.venv\Scripts\python.exe run_common_comparison.py --quick
.\.venv\Scripts\python.exe run_tree_comparison.py --quick
.\.venv\Scripts\python.exe run_reliability_stress.py --quick
.\.venv\Scripts\python.exe run_scaling.py --max-sites 500
.\.venv\Scripts\python.exe build_dashboard.py
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Remove `--quick` for the five-seed tree, 12-method and reliability runs. Then run
`python run_scale_stress.py`, `python run_operational_stress.py`, `python build_evidence_report.py`
and `python run_reserve_benchmark.py --common`. Run `python run_reliability_stress.py --scenarios sensor_recovery`
before generating the evidence report, and `python check_dashboard.py` after the dashboard build.
Run `python build_evidence_plots.py` after all full comparisons to refresh the enhanced figures.
Quick runs save distinct tables. Five-seed recurrent
models and stress matrices can take tens of minutes or longer on a laptop. Plan full runs before
the demo. The dashboard replays precomputed results and needs no live training.

The dashboard opens in dark mode. Use **Dark mode** in the sidebar to switch to light
mode. Charts, evidence plots, table cells and operator panels follow the selection;
it persists across scenario changes in the current session and does not change results.
`python check_dashboard.py` checks all four scenarios in both themes and verifies
that switching themes leaves the displayed metrics unchanged.

The supplied weather CSV has no verified year/location metadata. 2019 is assumed. Measured Indian
PV/load validation is a next milestone. The daily reserve LP is a retrospective benchmark,
while its uncertainty and monitoring are causal. A live pilot requires rolling planning.

## Problem
Grid operators must hold reserve for the gap between forecast and actual renewable output.
Better short-horizon forecasts can support better reserve decisions, but do not guarantee the same reliability — renewable sites
are owned by different operators who do not want to pool raw data, and some sites have bad
sensors. GridMesh lets sites train a shared forecaster **without routinely pooling raw data**,
offers **experimental aggregation safeguards for degraded sensors**, and turns forecast uncertainty into a
**physically constrained backup schedule** at one neighbourhood node.

FL itself is not our novelty. Our contribution is the reliability-aware integration:
trust-weighted aggregation, event-aware participation, and a scheduler that separates the expected
supply gap from forecast uncertainty. Every policy faces the same power, energy, and cost limits.

## Purpose and judging alignment
- **Efficiency:** event-aware participation reduces unnecessary client training and communication;
  the scheduler commits the least-cost feasible backup under the selected risk rule.
- **Sustainability:** improved forecasts can reduce unnecessary backup commitment. The evaluation
  measures this effect instead of assuming it.
- **Accessibility:** the operator sees the expected gap, uncertainty allowance, recommended backup,
  and a clear capacity warning. Software cost remains separate from hardware and energy.
- **Scalability:** the same interfaces run from 4 to 500 logical clients; sampled participation and
  repeatable neighbourhood coordinators avoid one unconstrained city-scale optimizer.

Target users are community energy operators, ESCOs, and DISCOM teams managing clusters that already
have solar, metering, and accessible dispatchable backup. GridMesh cannot create missing energy or
replace protection, islanding, or utility operating procedures.

## Quick start (macOS / Linux)
```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python audit.py                    # 36 automated correctness checks
python run_common_comparison.py    # enhanced 12-method, five-seed matrix (tens of minutes or longer)
python run_scaling.py              # 4..500 logical clients; see scale-data limits above
python build_dashboard.py          # pre-compute 4 dashboard scenarios (several minutes)
python check_dashboard.py          # all four scenario UI smoke tests
python -m streamlit run dashboard.py   # demo dashboard in the browser (Ctrl+C to stop)
```

## Commands
| Command | What it does |
|---|---|
| `python inspect_data.py` | Phase 1 dataset report |
| `python -m src.data.preprocessing` | virtual sites + split self-check (leakage assertions) |
| `python run_simulation.py --clients 4` | one run of all methods (baselines + FL) |
| `python run_simulation.py --methods fl --faulty-client C,D --fault-type stale` | faulty sites |
| `python run_simulation.py --methods fl --faulty-client D --fault-start 6` | fault appears mid-training (trust demo) |
| `python run_simulation.py --methods fl --dropout-rate 0.25` | random client dropout |
| `python run_simulation.py --split chronological` | seasonal-shift stress test (train Jan–Sep, test Nov–Dec) |
| `python run_simulation.py --clients 100 --rounds 10 --client-fraction 0.2 --methods smart_persistence,fl` | 100-site scale run |
| `python run_model_comparison.py [--quick]` | federated MLP / GRU / LSTM under one protocol |
| `python run_reserve_sim.py [--policy fixed20 \| nsigma \| empirical \| all] [--delta 0.05]` | constrained one-node backup schedule |
| `python run_experiments.py [--quick]` | full matrix over seeds → tables, `summary.json`, 13 plots, `RESULTS.md` |
| `python run_scaling.py` | 4..500-site payload/runtime stress; capped sampling and hierarchy |
| `python run_common_comparison.py [--quick]` | 12 forecasting methods with common held-out metrics and reserve outcomes |
| `python run_tree_comparison.py [--quick]` | Decision tree, local/pooled XGBoost and readable gain importance |
| `python run_reliability_stress.py [--quick]` | Paired dropout, faults, recovery and seasonal-shift scorecard |
| `python run_scale_stress.py` | Large logical-client fault/dropout/unavailable-neighbourhood tests |
| `python run_operational_stress.py` | Cloud shock, resource shortage and runtime forecast fallback |
| `python audit.py` | 36 correctness checks including tree fitting, fallback, rollback, recovery, hierarchy and causality |
| `python run_reserve_benchmark.py --common` | enhanced five-seed constrained reserve benchmark used in the pitch |
| `python check_dashboard.py` | Headless regression smoke test for all four dashboard scenarios |
| `python build_dashboard.py` + `python -m streamlit run dashboard.py` | dashboard: NORMAL / FAULTY CLIENT / CLIENT DROPOUT / WEATHER-REGIME SHIFT |

Options: `--fault-type stale (default) | target_noise | bias | feature_corruption`, `--fault-severity 2`,
`--faulty-frac 0.2` (random 20% of sites), `--seed`, `--rounds`. Sites are lettered A, B, C…
All tunable numbers live in `config.yaml`.

## Data — what is real, what is simulated
| Item | Status |
|---|---|
| Weather + irradiance (GHI/DNI/DHI, clear-sky, cloud type, temp…) | **Real** — NSRDB-format file, 1 year, 10-min, no gaps |
| Year | **Assumed** 2019 (file has Month/Day/Hour/Minute only) |
| `WindSpeed_Class` | Real label, **not used**: it is a class, not power |
| PV power target | **Modeled** from real GHI + temperature (simplified PVWatts, horizontal plane), p.u. of capacity |
| Sites (4 or 100) | **Simulated** heterogeneity from one real weather record: capacity, derate, temperature coefficient, sensor noise/bias/dropouts, meter noise, history length (younger plants have seen fewer seasons) |
| Faults | **Simulated** sensor/meter degradation — not cyber attacks |
| Demand at the node | **Synthetic** daily curve |
| Reserve / shortfall costs | **Simulation assumptions** (1 vs 20 cost units per MWh) |

## Using real data (the path to the full prototype)
Two switches in `config.yaml`, both exercised by `python audit.py`:
- **Measured power, one record:** `target.mode: column`, `target.column: <power column>`
  (optional `target.capacity`, otherwise the observed peak). Virtual sites are still simulated.
- **Many real plants in one CSV** (one row per plant per timestamp): also set
  `data.site_col: <site-ID column>`, ideally `target.capacity_col: <nameplate column>`, and
  `target.column_unit_mw: 0.001` if power is in kW. Each plant becomes one federated client and
  no simulated noise is added.

Every plant needs the weather columns the features use: `GHI, DNI, DHI, Clearsky GHI,
Solar Zenith Angle, Cloud Type, Temperature, Relative Humidity, Pressure, Dew Point,
Precipitable Water, Aerosol Optical Depth`. Satellite irradiance services such as NSRDB provide
these for any coordinates, so a deployment pairs each plant's SCADA power with satellite weather
for its location.

## Architecture

data/raw/*.csv ─► adapter ─► virtual_sites (simulated heterogeneity, faults)
│
preprocessing (lags, clear-sky features, blocked monthly split, per-site scaling)
│
┌────────── site A ── site B ── … ── site N (raw histories remain local by design)
│ local training (MLP / GRU / LSTM, correction to smart persistence)
│ report: parameters + n_samples + val error of received model + data-quality
▼
server: FedAvg | reliability-aware FedAvg | + event-aware participation
│ best-round selection from client-reported validation error
▼
forecasts + demand + asset limits ─► expected gap + uncertainty margin
                                      │
                                      ▼
                            constrained backup LP ─► schedule, capacity warning, cost

Code: `src/data` (loading, sites, features), `src/models` (MLP / GRU / LSTM), `src/federated` (client,
server, aggregators, selection), `src/reliability` (trust, drift), `src/reserve`,
`src/evaluation` (metrics, experiments, report), `src/visualization`.

## Algorithms
**Forecast target.** +30 min PV power. Each selectable backbone predicts the *correction* to smart
(clear-sky) persistence. MLP remains the default so the published repository results stay comparable;
`run_model_comparison.py` evaluates MLP, GRU and LSTM under the same federated protocol.

**Reliability-aware aggregation (our prototype formula, not a published method).** Each round,
clients score the model they *received* on their own recent validation data (e_k):

r_k = exp(-2 · max(0, e_k / median_j(e_j) − 1 − 0.25)) # 25% tolerance
trust_k = 0.5 · trust_k(prev) + 0.5 · r_k
quality_k = share of daytime sensor readings that changed (missing/frozen readings lower it)
weight_k = n_k · trust_k · quality_k (0 if trust_k < 0.2 → quarantined), then normalised

**Event-aware participation.** Drift if `e_now > mean + 2·std` of a site's last 5 errors and
at least 10% above that mean.
Drifting sites have priority within the configured client cap; stable ones train with probability 0.5 and otherwise send a
16-byte heartbeat (the error of the last model they received). A heartbeat only informs trust
if that model had been trained at least once.

**Model selection.** Like early stopping in the baselines: the server keeps the global model with
the lowest client-reported validation error (weighted like that method's aggregation).

**Uncertainty margin.** Node forecast/actual = Σ capacity_k · p.u._k. Error
`e = actual − forecast`. Fixed baseline: `m = 0.20 · demand`. Gaussian benchmark (Khaing,
Kannan & Rao, *Clean Energy* 2026, Theorem 4.3): `m = max(α·σ_e − μ_e, 0)`,
`α = Φ⁻¹(1−δ)`, using only errors observable when the forecast is issued. The empirical policy
uses a rolling observed tail quantile and avoids the Gaussian shape assumption.

**Constrained backup schedule.** The node first schedules grid import up to its assumed limit, then
computes `expected_gap = max(demand − grid − solar_forecast, 0)` and
`required_backup = expected_gap + margin`. A daily linear program minimizes backup commitment plus
planned uncovered requirement while enforcing backup power and energy limits. Real-time evaluation
deploys the scheduled backup against `max(demand − grid − solar_actual, 0)`. Inadequate capacity
produces an explicit warning rather than an unsupported availability claim. Demand, grid, backup,
and cost inputs are simulation assumptions.

## Experimental design (and why)
- **Blocked monthly split:** in every month the first 70% of days train, the next 15% validate, the
  last 15% test; no input window crosses a boundary. A single 70/15/15 cut of one year tests only
  Nov–Dec, a season barely in training — kept as a stress test (`--split chronological`).
- **Metrics on daytime targets only**, p.u. of installed capacity; per-site scaling from each
  site's own training data for every method.
- **Faults are present for the whole training run** (a site with a corrupted history). With
  best-round selection, a fault that starts mid-training can be escaped by keeping an earlier
  model, which would hide any damage; `--fault-start 6` is kept for the visual demo.
- **5 seeds**; seeds change the simulated sites, the model initialisation and the sampling.
  RESULTS.md also counts paired wins (same scenario, same seed) so the means are not the only evidence.
- **Younger plants:** a site with a shorter history holds only the most recent part of the training
  year, so sites have seen different seasons (non-IID). January is absent from the training data
  of all four default sites; every site is still validated and tested on all 12 months.
- Results differ in the 4th decimal between machines (PyTorch on Apple Silicon vs Linux).

## Current findings

### Indian-weather mentor validation (separate protocol)

Run `python run_mentor_validation.py` for 100 heterogeneous virtual rooftop sites,
five paired seeds and MLP/GRU/LSTM under the same 12-round federated protocol.
The archived NASA POWER Bengaluru record covers all 8,784 hourly observations of
2024 UTC. Targets are modeled PV, not measured plant output. The forecast horizon
is **60 minutes**, not the legacy 30-minute horizon. Hourly irradiation is converted
to interval-average irradiance without inventing sub-hourly observations.

This experiment fixes grid availability, demand, asset limits and the objective
coefficients across models. Its LP uses one-hour intervals and a constant
`1*q + 20*u` assumed-cost objective. `u` is uncovered **planned reserve requirement**,
not measured outages. Realized unserved energy and operating-cost score are
evaluated separately using actual modeled solar. Per-day scheduling remains a
retrospective benchmark, not online grid control.

Evidence is written to `outputs/mentor_validation/`, never mixed with legacy
results. Run `python verify_legacy_evidence.py` to re-fit the retained slide-7/9
legacy protocols, and `python -m unittest test_mentor_validation -v` for hourly
accounting and pairing regression tests. The existing `python audit.py` remains
the broad regression guard. All client processes are emulated in one Python
process. Training exchanges parameters/reports, but physical data isolation,
secure aggregation and measured Indian PV validation remain unimplemented.

Use [`docs/PITCH_NUMBERS.md`](docs/PITCH_NUMBERS.md) for pitch numbers and
[`outputs/RESULTS.md`](outputs/RESULTS.md) for the complete model, reliability, scale and reserve
tables. Those reports derive from completed frozen-protocol experiments. Failed targets remain
visible. Legacy `run_experiments.py` outputs use an older experiment matrix and overwrite the
results report, so run `build_evidence_report.py` afterward to restore the current report.
Do not mix legacy two-transfer communication values with the new three-transfer accounting.

## Roadmap: from this simulation to the 100-site prototype
| Step | What changes | Where in the code |
|---|---|---|
| Real plants | SCADA power + per-site satellite weather, one client per plant | `config.yaml` (`site_col`, `target`), `real_sites()` in `src/data/virtual_sites.py` |
| Real network | sites as separate processes (gRPC / MQTT, e.g. the Flower framework) | `src/federated/client.py` and `server.py` keep their `fit / evaluate / weights` interface |
| Privacy | secure aggregation (pairwise masking), optional differential privacy | aggregation step in `src/federated/fedavg.py` |
| Model selection | validate MLP / GRU / LSTM and personalised layers with measured sites | `src/models/forecasting_model.py` |
| Calibrated reserve | select Gaussian or empirical windows on validation data | `src/reserve/reserve_policy.py` |
| Asset-constrained schedule | replace assumed grid / backup limits with operator data | `src/reserve/simulator.py` |
| Drift handling | validate fine-tuning and safe fallback under unseen seasons | `src/reliability/drift.py`, `server.py` |
| Live operation | forecasts every 10 min streamed into the dashboard | `dashboard.py` |

Run `python audit.py` after every change — it is the regression guard.

## Current operator demo and credible submission extension

From the repository root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe build_advisory_demo.py
.venv\Scripts\python.exe -m streamlit run dashboard.py
```

Select **Deterministic advisory demo** in the sidebar. It does not require training,
network access or legacy dashboard replay files. Normal, cloud, insufficient backup,
stale/missing input and degraded-site scenarios expose forecasts, power/energy,
uncovered reserve and separate synthetic actual outcomes. Zero-solar fallback is a
labelled conservative planning bound. Human approval/physical dispatch remain outside
the application. The unchecked dashboard view is the separate legacy4-site replay.

```powershell
.venv\Scripts\python.exe -m unittest test_evidence_improvement test_mentor_validation test_operating_improvements -q
.venv\Scripts\python.exe audit.py
.venv\Scripts\python.exe check_operator_demo.py
.venv\Scripts\python.exe check_dashboard.py
.venv\Scripts\python.exe verify_evidence_improvement.py
.venv\Scripts\python.exe build_full_service_costs.py
.venv\Scripts\python.exe build_hackathon_findings.py
```

`check_dashboard.py` also requires the existing replay files; generate them using
`build_dashboard.py` if absent. See [implemented findings](outputs/hackathon_completion/FINDINGS.md),
[proposed pilot/O&M](docs/PILOT_AND_OPERATIONS.md), [claim boundaries](docs/CLAIMS_REGISTER.md)
and editable [itemised costs](configs/affordability_costs.yaml). Historical known-subtotal
cost files and old submission packages are superseded by `outputs/affordability_v1`.
Missing quotes/taxes keep all-in price and break-even TBD. No sourced Bengaluru installed
price, asset agreement, field reliability or willingness-to-pay is established.

Reproduce new studies into NEW directories, never overwrite completed evidence:

```powershell
.venv\Scripts\python.exe run_india_failure_checks.py --out outputs/reproduced_faults --seeds 42 43 44
.venv\Scripts\python.exe run_operating_sensitivity.py --out outputs/reproduced_operating
```

Completed final sensitivity is `outputs/operating_sensitivity_v3`; v1 incomplete and v2
superseded after correcting night-only stress-event alignment. Frozen original source
hashes, preregistration and validation policy freeze are included. New guard is experimental;
default nominal planner unchanged. No consistent current-India weighting accuracy benefit.

## Latest submission artifacts

- [Final 12-slide PowerPoint](docs/submission/GridMesh_Final_Submission.pptx)
- [Submission PDF](docs/submission/GridMesh_Final_Submission.pdf)
- [Actual dashboard screenshots](docs/submission/GridMesh_UI_Screenshots.zip)

The latest deck includes the implemented advisory UI, simulation evidence, a
prototype-build roadmap, and the itemised approximately INR15,268/month covered
service estimate. The funding example is conditional, not an approved subsidy.
All-in costs, willingness to pay and field reliability remain unvalidated.
Older submission files are retained as historical artifacts, not the current deck.

## Limitations / honesty notes
- Raw training data stays local; model parameters and a few summary numbers are exchanged.
  **No secure aggregation or differential privacy** is implemented — no privacy guarantee is claimed.
- One weather record shared by all simulated sites (co-located sites at one node); real fleets
  have weather diversity.
- PV power is modeled, not measured; demand is synthetic; costs are assumptions.
- Chronological selective participation trades +2.72% relative mean RMSE for 45.55% fewer
  modeled protocol bytes. Older approximately 1%/half-communication values belong to
  separate reference protocols; none is measured physical network traffic.
