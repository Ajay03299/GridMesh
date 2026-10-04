# GridMesh: improvements and pilot gaps

This earlier report describes the initial chronological evidence pass. The subsequent
hackathon extension is in `outputs/hackathon_completion/FINDINGS.md`; final deck v34
reflects that verified extension. Legacy and mentor-validation experiments remain frozen.

## Weakness → action checklist

| Weakness | Status | Action / boundary |
|---|---|---|
| Backup savings reported without reliability loss | Resolved as an evidence issue; not a guaranteed gain | Recompute frozen test metrics; report lost reliability. Validation-only selector must match fixed reserve on both availability and ENS in every validation seed. Test may still disappoint. |
| Monthly blocks/full-day planning presented like real operation | Partially resolved | Preserve reference; add separate chronological, one-interval causal benchmark. Do not claim direct improvement across changed protocols. |
| Missing current-experiment baselines | Resolved in software; see findings for outcomes | Persistence, smart persistence, local MLP, FedAvg, reliability-weighted and selective-update FL; paired data/asset checks, documented unequal compute budgets. |
| Planned reserve confused with realized ENS | Resolved in new workflow | q = scheduled backup; planned_gap = uncovered requirement; ENS evaluated separately using actual solar. Legacy planning_unserved_cost_per_mwh names an uncovered-reserve coefficient, not realized ENS. |
| Zero/constrained advice unclear | Resolved for advisory prototype | Power/energy limits, requirement, uncovered reserve, freshness and reasons; deterministic cloud-shock/exhausted-energy demo. |
| Backup technology/access unspecified | Externally blocked | Generic asset only; owner/supplier agreement, technology, replenishment and safety procedures required. |
| ₹46.50 appears all-in/validated | Partially resolved | Editable known-subtotal model; unknown labour, maintenance, connectivity, energy and assets remain TBD. Participation/cost-driver sensitivity, no WTP validation. |
| Robustness/privacy/network claims too broad | Partially resolved | Failure regression tests; separately labeled legacy stress results. No formal privacy, physical-network measurement or measured uptime. |
| Reproducibility | Resolved for new experiment/demo | Split assertions, paired inputs, pre-test policy freeze/hash, configs, per-seed/site results, provenance and manifest. Completed experiments cannot be overwritten. |
| Field reliability, all-in affordability, ownership | Externally blocked | Measured pilot, quotes, interviews and responsibilities/approval needed. |

## Reproduce (Windows, repository root)

```powershell
.\.venv\Scripts\python.exe -m unittest test_mentor_validation test_evidence_improvement -v
.\.venv\Scripts\python.exe audit.py
.\.venv\Scripts\python.exe run_evidence_improvement.py
.\.venv\Scripts\python.exe run_india_failure_checks.py
.\.venv\Scripts\python.exe verify_evidence_improvement.py
.\.venv\Scripts\python.exe report_evidence_improvement.py
.\.venv\Scripts\python.exe build_advisory_demo.py
.\.venv\Scripts\python.exe -m streamlit run dashboard.py
.\.venv\Scripts\python.exe check_dashboard.py
```

Select **Deterministic advisory demo** in the dashboard. No trained replay data required for that view.
Move the slider: cloudy forecast → power-constrained advice → stale warning → exhausted energy.
It is synthetic advice, not equipment control. The old four-scenario view remains a retrospective
historical replay. Invalid/nonfinite inputs are rejected rather than silently changed to zero.

For another complete reproduction use `run_evidence_improvement.py --out outputs/evidence_improvement_reproduction`.
The report reads the default `outputs/evidence_improvement`. Five seeds, 100 virtual sites,
12 FL rounds, up to 20 clients/round; local-only 12 full epochs/site. Budgets are documented,
not equal total compute. Basic FedAvg retains the shared safety gates: a weighting baseline,
not entirely unguarded FL. No new complex model or dependency was introduced.

`run_india_failure_checks.py` is a bounded one-seed stress smoke using 20 stale training sites and
50% random client dropout; clean targets are retained for evaluation. It uses the clean validation
policy freeze without re-tuning. This is not five-seed robustness evidence or uptime validation.

Candidate policies are predeclared in `configs/evidence_improvement.yaml`. Select on validation
availability AND ENS per seed, then minimum backup, with fixed reserve as fallback. Freeze before
operational test evaluation; never edit candidates based on test results. Earlier observed test
residuals can adapt margins; current/future solar cannot. No tail/service guarantee is claimed.

## Temporal and scientific audit

- Old monthly blocks are disjoint but final models train on later months than early test targets.
  This is an offline seasonal benchmark, not prospective year-long operation.
- New chronological targets satisfy max(train) < min(validation) < min(test). Standardizers use
  training only; supervised windows cannot cross split boundaries. Future covariates are calendar/
  astronomy references, not future observed weather. Validation selects checkpoints and policies,
  so it is a selection set rather than unbiased performance evidence.
- Old full-day LP sees future forecast/margin sequences. New one-interval LP has an analytic solution
  under constant uncovered penalty > backup cost: min(requirement, power, remaining energy/duration).
  It is myopic, not a claimed superior day optimizer. Split/planner change defines a new experiment.
- Synthetic target demand and fixed grid availability are assumed known at decision time (ideal load
  forecast). A pilot needs measured load/grid signals and load uncertainty. Scheduled backup consumes
  the energy budget even if unused. Night adequacy is outside the daytime evaluation.
- NASA POWER is gridded weather, not measured generation. One co-located weather record, modeled PV,
  virtual sites, assumed generic backup, no geographic holdout or measured outage hours. Traffic is
  modeled protocol bytes, not measured WAN traffic. FL is not formal privacy.
- Observation rows are assumed available at the modeled issue time; NASA publication latency and
  hourly-bin start/end conventions are not a validated live sensor feed. A pilot must timestamp
  interval closure, actual ingestion and forecast issuance before asserting decision-time availability.

## Ownership and economics

Proposed owner: neighbourhood association/ESCO, subject to agreement. Local operator checks data/assets
and approves/rejects advice; qualified asset maintainer handles safety/maintenance. DISCOM receives
forecast, uncertainty, expected shortfall, reserve requirement and health signals. Owner access,
data permissions, actual interfaces and DISCOM approval remain unconfirmed. Paying households share
software cost; asset availability/energy charges need separate agreements.

Current correction: edit `configs/affordability_costs.yaml` and run
`build_full_service_costs.py`. Unsupported₹4,650/46.50/93 allocation is withdrawn.
Published hosting/connectivity rates and explicit paid-task assumptions now produce
itemised covered costs, low/base/high,50/100 participation and shortfall sensitivities.
See docs/AFFORDABILITY.md and outputs/affordability_v1. Complete economics remainTBD:
retrofit, owner access and actual consumption change are not silently zero. New assets
select capital recovery OR financing. No abstract score or scheduled-MWh-to-INR saving.

## Pilot prerequisites and success criteria (proposed, not achieved)

1. Obtain synchronized measured solar, neighbourhood load and grid availability, metering permissions,
   named operator, maintainer, asset owner and DISCOM contact. Agree freshness/safety escalation.
2. Confirm backup technology, usable power/energy, efficiency, replenishment, availability and dispatch
   latency; obtain installed sensor/gateway/asset quotes and operating-cost responsibility.
3. Start shadow/advisory mode. Pre-register paired intervals, fixed baseline, forecast horizon, supply
   availability/ENS definitions, sample count, cost scope and acceptable bounds before evaluation.
4. Success requires no worse measured availability/ENS with lower scheduled backup or verified cost;
   report variability/failures/overrides. Also verify asset-limit adherence, stale/invalid warnings,
   traceable records and operator usability; no invented timed-usability metric.
5. Interview households, validate participation and total cost. Replicate only after independent
   feeder/weather tests and safety approval. No emissions/loss reduction or Schneider partnership claim.

## Claims for future PPT updates

Use completed `outputs/evidence_improvement/FINDINGS.md` and per-seed CSVs: chronological evaluation,
explicit simple baselines, validation-only selection, constrained causal advice, planned-vs-realized
metrics, and honest unknown economics. Whether backup savings preserve reliability and whether FL/
selective updates earn their complexity are results, not promises. Keep legacy 4-site +30-minute,
old blocked Bengaluru, and new chronological Bengaluru experiments distinct. Field reliability,
affordability, geographic transfer and willingness to pay remain unvalidated. The PPT was not modified.
