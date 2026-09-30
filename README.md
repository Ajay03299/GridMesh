# GridMesh — Reliability-aware Federated Renewable Forecasting

Hackathon prototype · Yuva Yodha Energy Tech Hackathon · Grid Reliability & Renewable Intermittency track.

**Status:** Phases 1–8 done (data → baselines → FedAvg → reliability-aware FL → faults/dropout).
Next: severity sweep + seeds, reserve simulator (Phase 9), plots (10), full README (11).

## Idea
Each PV site keeps its data locally and trains a small forecaster. Sites collaborate via
Federated Learning. We compare vanilla FedAvg with a **reliability-aware** aggregation that
down-weights or quarantines sites whose data looks degraded. A reserve scheduler at one
aggregation node will turn forecast uncertainty into a reserve requirement (n-sigma rule,
inspired by Khaing, Kannan & Rao, *Clean Energy* 2026). FL itself is not our novelty; the
reliability-aware integration and its honest evaluation are.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Commands
```bash
python inspect_data.py                          # Phase 1: dataset report
python -m src.data.preprocessing                # Phase 2: sites + split self-check
python -m src.models.forecasting_model          # Phase 3: one local model (site A)
python run_simulation.py --clients 4 --methods baselines          # Phases 4-5
python run_simulation.py --clients 4 --rounds 20 --methods fl     # Phases 6-7
python run_simulation.py --clients 4 --rounds 20 --methods fl --faulty-client D --fault-type stale
python run_simulation.py --clients 4 --rounds 20 --methods fl --dropout-rate 0.25
python run_simulation.py --clients 4 --split chronological        # seasonal-shift stress test
python run_simulation.py --clients 4            # everything (baselines + FL), ~75 s
```
Outputs: `outputs/tables/results_*.csv` (metrics), `outputs/logs/clients_*.csv`
(per-round weight / trust / drift / participation per site), `outputs/metrics/run_*.json`
(config + seed + site parameters + results, for reproducibility).
Fault types: `feature_corruption`, `target_noise`, `stale`, `bias`. `--faulty-client 3` = `D`.

## Data — what is real, what is simulated
| Item | Status |
|---|---|
| Weather + irradiance (GHI/DNI/DHI, clear-sky, cloud type, temp…) | **Real** — NSRDB-format file, 1 year, 10-min, 0 gaps |
| Year | **Assumed** 2019 (file has Month/Day/Hour/Minute only) |
| `WindSpeed_Class` (5 balanced classes) | Real label, **not used** as target: it is not power |
| PV power target | **Modeled** from real GHI + temperature (simplified PVWatts, horizontal plane) — not measured |
| 4 / 100 virtual sites | **Simulated** heterogeneity: capacity, derate, temp. coeff., sensor noise/bias/dropouts, meter noise, history length. All sites share the one real weather record |
| Faults | **Simulated** sensor/meter degradation (no cyber attacks) |

## Key design decisions (and why)
1. **Solar PV regression, not wind classification.** The file contains real irradiance, so a
   physically-grounded continuous power target exists; a wind *class* cannot feed a reserve model.
2. **Target = correction to smart persistence** (clear-sky persistence). Chosen on the
   *validation* set: beat direct prediction on all 4 sites; removed a seasonal over-forecast bias.
3. **Blocked monthly split** (inside each month: first 70% days train, 15% val, 15% test; no
   window crosses a boundary). A single global 70/15/15 cut on a 1-year file tests only on
   Nov–Dec, a season barely in training. The global cut is kept as a *stress test* (`--split chronological`).
4. Every site standardises with **its own** training statistics, for every method.
5. Metrics on **daytime targets only**, in p.u. of installed capacity (0.05 = 5% of capacity).

## Reliability-aware aggregation (our prototype formula)
Each round, clients score the *received global model* on their own recent validation data (e_k):
```
rel_k   = e_k / median_j(e_j)
r_k     = exp(-2.0 * max(0, rel_k - 1 - 0.25))        # tolerance 25%, sharpness 2
trust_k = 0.5 * trust_k(prev) + 0.5 * r_k              # smoothed
quality_k = fresh-reading rate (share of daytime GHI readings that changed; missing/frozen lower it)
weight_k  = n_k * trust_k * quality_k   (0 if trust_k < 0.2 -> quarantined)   then normalised
```
Event-aware participation: drift if `e_now > mean + 2*std` of the site's last 5 errors;
drifting sites always join; stable ones join with prob 0.5 and otherwise send a 16-byte heartbeat.
All constants live in `config.yaml`.

## Findings so far (4 sites, seed 42, measured — rerun the commands above to reproduce)
Test = daytime, all months, p.u. of capacity.

| Method | MAE | RMSE | Comm (MB, 20 rounds) |
|---|---|---|---|
| Persistence | 0.0680 | 0.0918 | – |
| Smart persistence | 0.0364 | 0.0743 | – |
| Local-only | 0.0365 | 0.0651 | – |
| Centralized (pooled data) | 0.0370 | 0.0646 | – |
| FedAvg | 0.0357 | 0.0656 | 5.08 |
| Reliability-aware FedAvg | 0.0353 | 0.0658 | 5.08 |

- All learned models beat smart persistence on RMSE by ~12%; MAE is about tied.
- No faults: reliability-aware weights ≈ FedAvg weights (as intended); accuracy tied.
- Faulty site D: the mechanism detects `target_noise` and `stale` faults and drives D's weight
  to 0 (quarantine by round 8), partially down-weights `bias` (0.22 → 0.15), and does **not**
  detect `feature_corruption`. **But** one bad site out of 4 barely hurts vanilla FedAvg here,
  so the accuracy differences are within noise. A severity sweep over several seeds is next.
- Event-aware participation cuts communication by ~45–55% with no accuracy loss (slightly lower
  RMSE, likely because fewer local updates reduce over-fitting — not claimed as a benefit yet).
- 25% client dropout: training continues, RMSE 0.0657 (vs 0.0656).
- Seasonal-shift stress test (train Jan–Sep, test Nov–Dec): learned models over-forecast
  (+0.02 p.u.) and lose to smart persistence (RMSE 0.052 vs 0.043). Real risk; motivates drift handling.

## Honesty notes
- Raw training data stays local; only model parameters and a few summary numbers are exchanged.
  **No secure aggregation / differential privacy is implemented**, so no privacy guarantee is claimed.
- The trust formula is our prototype mechanism, not a published canonical method.
- Cost coefficients and demand (reserve phase) will be simulation assumptions, labelled as such.
