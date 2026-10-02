# GridMesh hackathon alignment

## Purpose

Small solar portfolios and neighbourhood feeders need dependable renewable power, but their data is
fragmented, sensors are imperfect, and overcommitting backup is expensive. GridMesh lets sites learn
a shared 30-minute-ahead forecast without pooling raw operational data, detects unreliable client
updates, and converts uncertainty into a resource-constrained feeder backup schedule.

## Evaluation criteria

| Criterion | What GridMesh demonstrates | Evidence in the repository |
|---|---|---|
| Relevance | Directly addresses renewable intermittency and local grid reliability | Forecast-to-reserve pipeline and one-node scheduler |
| Impact | Supports a 4-to-100-site simulation and measured multi-site input path | `run_scale_test.py`, `audit.py`, configuration-driven clients |
| Innovation | Reliability-aware aggregation + event-aware participation + causal uncertainty margins + constrained scheduling | `src/reliability`, `src/federated`, `src/reserve` |
| Complexity | End-to-end data, three model families, FL, fault injection, drift logic, linear optimization, evaluation, dashboard | Source tree and 24 automated checks |
| Implementation | Reproducible commands generate tables, plots, metrics, and dashboard data | `run_experiments.py`, `build_dashboard.py`, `dashboard.py` |
| Clarity | Dashboard decomposes each action into forecast, demand, expected gap, uncertainty, and backup | Streamlit operator view |

## Schneider purpose themes

- **Efficiency:** event-aware participation reduces communication; constrained scheduling avoids
  treating backup as unlimited and exposes the cost/reliability trade-off.
- **Sustainability:** more dependable renewable forecasts can reduce avoidable fossil backup and
  renewable curtailment; these are intended outcomes, not yet field-verified impact claims.
- **Accessibility:** the lightweight Python stack can run on ordinary hardware; raw site histories
  remain local; the interface uses operator language and explicit warnings.
- **Scalability:** the same coordinator logic supports neighbourhoods, campuses, community solar,
  and feeder portfolios; client sampling and heartbeat messages limit network growth.

## Positioning relative to Schneider Electric

GridMesh is not a replacement for EcoStruxure DERMS, Microgrid Advisor, or Microgrid Operation.
Those platforms provide broader production orchestration, monitoring, and lifecycle capabilities.
GridMesh is a complementary, research-stage module for a narrower gap: privacy-preserving
cross-owner renewable forecasting, reliability scoring of site updates, and transparent
forecast-to-reserve recommendations for smaller portfolios. A mature version could publish
forecasts, confidence, and recommended flexibility through standard interfaces into a utility or
microgrid management platform.

## Demo story

1. Show four sites forecasting locally; no raw histories leave the sites.
2. Inject a stale or noisy sensor and show its trust and aggregation weight fall.
3. Compare MLP, GRU, and LSTM under one protocol; select the compact model from measured evidence.
4. Aggregate forecasts at one node and show demand, grid import, expected gap, uncertainty margin,
   scheduled backup, and any capacity warning.
5. Change the risk target or asset limits and show the reliability/cost trade-off.

## Honest scope

This is a decision-support prototype, not a field-certified control system. Secure aggregation,
differential privacy, AC power-flow constraints, measured feeder demand, live SCADA integration,
cybersecurity hardening, and a real pilot are roadmap items. Keeping these boundaries explicit makes
the first-round proposal more credible and gives the implementation roadmap concrete milestones.
