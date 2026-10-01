# GridMesh — Reliability-aware Federated Renewable Forecasting

Hackathon prototype · Yuva Yodha Energy Tech Hackathon · Grid Reliability & Renewable Intermittency track.

**Measured results: [`outputs/RESULTS.md`](outputs/RESULTS.md)** (auto-generated, every number from the code).

## Problem
Grid operators must hold reserve for the gap between forecast and actual renewable output.
Better short-horizon forecasts mean less reserve for the same reliability — but renewable sites
are owned by different operators who do not want to pool raw data, and some sites have bad
sensors. GridMesh lets sites train a shared forecaster **without sharing raw data** (Federated
Learning), makes the aggregation **robust to sites with degraded data**, and turns forecast
uncertainty into a **reserve requirement** at an aggregation node.

FL itself is not our novelty. Our contribution is the reliability-aware integration
(trust-weighted aggregation + event-aware participation + reserve sizing) and its honest evaluation.

## Quick start (macOS / Linux)
```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_experiments.py          # full experiment matrix, 3 seeds, plots + RESULTS.md (~3-5 min)
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
| `python run_reserve_sim.py [--policy fixed20 \| nsigma] [--delta 0.05]` | reserve node on the last run's forecasts |
| `python run_experiments.py [--quick]` | full matrix over seeds → tables, `summary.json`, 13 plots, `RESULTS.md` |

Options: `--fault-type feature_corruption | target_noise | stale | bias`, `--fault-severity 2`,
`--faulty-frac 0.2` (random 20% of sites), `--seed`, `--rounds`. Sites are lettered A, B, C…
All tunable numbers live in `config.yaml`.

## Data — what is real, what is simulated
| Item | Status |
|---|---|
| Weather + irradiance (GHI/DNI/DHI, clear-sky, cloud type, temp…) | **Real** — NSRDB-format file, 1 year, 10-min, no gaps |
| Year | **Assumed** 2019 (file has Month/Day/Hour/Minute only) |
| `WindSpeed_Class` | Real label, **not used**: it is a class, not power |
| PV power target | **Modeled** from real GHI + temperature (simplified PVWatts, horizontal plane), p.u. of capacity |
| Sites (4 or 100) | **Simulated** heterogeneity from one real weather record: capacity, derate, temperature coefficient, sensor noise/bias/dropouts, meter noise, history length |
| Faults | **Simulated** sensor/meter degradation — not cyber attacks |
| Demand at the node | **Synthetic** daily curve |
| Reserve / shortfall costs | **Simulation assumptions** (1 vs 20 cost units per MWh) |

**Use your own data:** put a CSV in `data/raw/`, set `data.path` in `config.yaml`. With measured
power, set `target.mode: column` and `target.column: <name>`; feature lists are in `features:`.

## Architecture
```
data/raw/*.csv ─► adapter ─► virtual_sites (simulated heterogeneity, faults)
                              │
                 preprocessing (lags, clear-sky features, blocked monthly split, per-site scaling)
                              │
     ┌────────── site A ── site B ── … ── site N   (raw data never leaves a site)
     │  local training (MLP 64-64, learns correction to smart persistence)
     │  report: parameters + n_samples + val error of received model + data-quality
     ▼
  server: FedAvg | reliability-aware FedAvg | + event-aware participation
     │  best-round selection from client-reported validation error
     ▼
  forecasts ─► aggregation node: fixed 20% vs n-sigma reserve ─► shortfall, cost
```
Code: `src/data` (loading, sites, features), `src/models` (MLP), `src/federated` (client,
server, aggregators, selection), `src/reliability` (trust, drift), `src/reserve`,
`src/evaluation` (metrics, experiments, report), `src/visualization`.

## Algorithms
**Forecast target.** +30 min PV power. The MLP predicts the *correction* to smart (clear-sky)
persistence — chosen on validation data, it beat direct prediction on every site.

**Reliability-aware aggregation (our prototype formula, not a published method).** Each round,
clients score the model they *received* on their own recent validation data (e_k):
```
r_k       = exp(-2 · max(0, e_k / median_j(e_j) − 1 − 0.25))     # 25% tolerance
trust_k   = 0.5 · trust_k(prev) + 0.5 · r_k
quality_k = share of daytime sensor readings that changed (missing/frozen readings lower it)
weight_k  = n_k · trust_k · quality_k     (0 if trust_k < 0.2 → quarantined), then normalised
```
**Event-aware participation.** Drift if `e_now > mean + 2·std` of a site's last 5 errors.
Drifting sites always train; stable ones train with probability 0.5 and otherwise send a
16-byte heartbeat.

**Model selection.** Like early stopping in the baselines: the server keeps the global model with
the lowest client-reported validation error (weighted like that method's aggregation).

**Reserve.** Node forecast/actual = Σ capacity_k · p.u._k. Error `e = actual − forecast`.
Fixed: `r = 0.20 · forecast`. n-sigma (inspired by Khaing, Kannan & Rao, *Clean Energy* 2026):
`r = max(α·σ_e − μ_e, 0)`, `α = Φ⁻¹(1−δ)`, with μ_e, σ_e from the last 6 h of errors already
observable when the forecast is issued.

## Experimental design (and why)
- **Blocked monthly split:** in every month the first 70% of days train, the next 15% validate, the
  last 15% test; no input window crosses a boundary. A single 70/15/15 cut of one year tests only
  Nov–Dec, a season barely in training — kept as a stress test (`--split chronological`).
- **Metrics on daytime targets only**, p.u. of installed capacity; per-site scaling from each
  site's own training data for every method.
- **Faults are present for the whole training run** (a site with a corrupted history). With
  best-round selection, a fault that starts mid-training can be escaped by keeping an earlier
  model, which would hide any damage; `--fault-start 6` is kept for the visual demo.
- **3 seeds**; seeds change the simulated sites, the model initialisation and the sampling.
- Results differ in the 4th decimal between machines (PyTorch on Apple Silicon vs Linux).

## Headline findings (3 seeds, 4 sites — see RESULTS.md for exact values)
- All FL variants ≈ centralized (RMSE ≈ 0.064 p.u.) and beat local-only (≈ 0.067), with the
  largest gain on the weakest site (worst-site ≈ 0.067 vs 0.073). FL beats smart persistence on
  RMSE by ~13% (local-only ~9%); MAE is roughly tied.
- **Faulty sites:** vanilla FedAvg's healthy sites degrade (+2.7% on average, +9% with two
  stale-sensor sites, growing with fault severity); reliability-aware FedAvg stays at the clean level
  (≈ 0%). Feature corruption and multiplicative meter bias barely hurt either method.
- **Event-aware participation** cuts communication by ~45% at a small, within-noise accuracy cost.
- **Dropout** up to 50%: training continues; RMSE changes < 1%.
- **Reserve:** n-sigma δ = 0.05 holds about the same reserve as fixed 20% with ~25% less energy
  not served; δ = 0.10 holds ~20% less reserve with slightly less energy not served. n-sigma
  **misses its own availability target** (e.g. 91.9% vs 95%) because errors are heavier-tailed than
  Gaussian and the rolling window lags cloud ramps.
- **100 sites** (20% sampled per round, 10 rounds): runs in under a minute on a MacBook, ~3 GB RAM;
  event-aware participation saves ~42% communication. With 20% of sites faulty the
  reliability-aware advantage is small (single seed) — random client sampling already dilutes bad sites.
- **Seasonal shift** (train Jan–Sep, test Nov–Dec): learned models over-forecast and lose to smart
  persistence — a real limitation and the motivation for drift handling.

## Limitations / honesty notes
- Raw training data stays local; model parameters and a few summary numbers are exchanged.
  **No secure aggregation or differential privacy** is implemented — no privacy guarantee is claimed.
- One weather record shared by all simulated sites (co-located sites at one node); real fleets
  have weather diversity.
- PV power is modeled, not measured; demand is synthetic; costs are assumptions.
