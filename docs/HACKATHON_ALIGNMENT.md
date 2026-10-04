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
| Impact | 4-to-500 logical-client scale evidence and a measured multi-site adapter | `run_scaling.py`, `audit.py`; repeated-reference limits disclosed |
| Innovation | Model screening, trust, event sampling, hierarchy, fallback/rollback, causal margin health and constrained scheduling | `src/models`, `src/reliability`, `src/federated`, `src/reserve` |
| Complexity | Working data-to-forecast-to-reserve pipeline with failure handling and operator replay | Source tree and 36 automated checks |
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
GridMesh is a complementary, research-stage module for a narrower gap: privacy-conscious
cross-owner renewable forecasting, reliability scoring of site updates, and transparent
forecast-to-reserve recommendations for smaller portfolios. A mature version could publish
forecasts, confidence, and recommended flexibility through standard interfaces into a utility or
microgrid management platform.

Schneider's public descriptions already overlap with forecasting and optimization. We do not
claim Schneider lacks XGBoost, FL, hierarchy or equivalent internal capabilities. Public sources:
[utility solutions](https://www.se.com/ww/en/work/solutions/electric-utilities/),
[Microgrid Advisor](https://www.se.com/us/en/work/products/explore/ecostruxure-microgrid-advisor/),
[Microgrid Operation](https://www.se.com/us/en/product-range/65897-ecostruxure-microgrid-operation/),
[DERMS](https://www.se.com/us/en/product-range/89571422-ecostruxure-derms/).
Future integration requires measured-data validation, interfaces, security review and approval.

## Affordability and community operations

The proposed owner is a community operator/ESCO with a trained local operator and DISCOM escalation.
The unsupported INR4,650 software allocation is withdrawn. The itemised model in
configs/affordability_costs.yaml estimates covered added service at INR6,996–40,874/month,
with a15,268 base. At100/50 payers base covered recovery is about INR153/305 per home.
Unquoted retrofit/access and energy changes remain excluded. No Schneider price
comparison or cash saving claimed. Complete economics and WTP remain unresolved.

## Strong submission, with evidence boundaries

The mentor's approximate 75% prediction / 25% single-node reserve allocation remains the scope.
The operator journey precedes the equations in the deck. The key technical question remains
whether better forecasts improve feasible reserve outcomes under limited resources. Source of
truth is the generated current evidence report. Model choice uses validation, not test reserve
outcomes. Failed reliability targets and adverse guarded-policy outcomes stay visible.

The current daily LP is a retrospective schedule benchmark. Real-time rolling dispatch, Indian
data provenance, measured feeder demand, geographic generalization and verified emissions/curtailment
effects are pilot milestones. A credible proposal cannot guarantee selection or winning.

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
