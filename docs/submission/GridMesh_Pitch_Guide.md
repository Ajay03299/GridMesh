# GridMesh round-one pitch guide

## The story in one sentence

GridMesh lets solar sites improve one shared forecast without pooling raw histories, detects
unreliable updates, and converts forecast risk into a backup schedule that respects real asset
limits.

## Why the deck is structured this way

1. **Purpose first:** open with the reliability and affordability decision, not the technology.
2. **Operator journey:** make the idea understandable before introducing federated learning.
3. **Architecture:** show the mentor’s 75% forecasting and 25% reserve split.
4. **Technical evidence:** prove that the design is implemented, fault-aware and scalable.
5. **Research basis:** distinguish the paper’s uncertainty margin from the full reserve schedule.
6. **Optimization:** show the actual linear program, power limit, energy limit and capacity warning.
7. **Measured prototype result:** give the five-seed improvement with simulation labels visible.
8. **Judging alignment:** connect the evidence to efficiency, sustainability, accessibility and scale.
9. **Schneider positioning:** present GridMesh as a focused module beside existing platforms.
10. **Closing ask:** advance to the prototype round and validate on measured feeder data.

## Current evidence

Use `docs/PITCH_NUMBERS.md` and the final deck for current numbers. Older submission values
are historical and must not be mixed with the enhanced benchmark. Explain local XGBoost versus
smart persistence, reliability-aware versus standard FedAvg under faults, and forecast-to-reserve
outcomes under the same resource limits. Use the accuracy change with communication savings.

Weather is real, PV is modelled, sites/faults are simulated, demand is synthetic, and asset/cost
parameters are assumptions. Larger tests reuse compact datasets. Indian provenance is unverified.

## How to explain the reserve calculation

Say: “The research formula gives the safety margin for forecast error. It does not give the whole
backup schedule. We first calculate the expected gap between demand, grid import and forecast solar.
Then we add the causal uncertainty margin. Finally, a daily linear program schedules backup within
its MW power limit and MWh energy budget. If the assets are insufficient, we show a capacity warning.”

For the Gaussian policy:

`margin = max(Phi^-1(1-delta) * rolling_error_std - rolling_error_mean, 0)`

The residual statistics use only errors already observable at the forecast issue time. The empirical
policy uses a rolling tail quantile as a non-Gaussian benchmark.

## Likely judge questions

**Is federated learning itself novel?**  
No. The contribution is the combined pipeline: reliability-weighted client updates, event-aware
participation, causal reserve margins and constrained scheduling evaluated on operational outcomes.

**Why use MLP instead of GRU or LSTM?**  
All three are implemented under one FL protocol. The five-seed table reports validation/test
errors, runtimes and payload sizes. MLP remains the compact implementation. Any accuracy advantage
from a recurrent model is reported, and pilot selection will use validation data.

**Does raw data never leave a site?**  
Raw histories remain local in the prototype. Model updates still require protection; secure
aggregation, authentication and privacy review are pilot requirements.

**Does 95% mean 95% feeder uptime?**  
No. It is a nominal uncertainty-margin target under statistical assumptions. Feeder availability
also depends on grid and backup limits. GridMesh reports both realized availability and capacity gaps.

**How is this different from Schneider products?**  
Schneider’s DERMS and microgrid products cover broader production coordination and control. GridMesh
focuses on cross-owner federated forecasting, unreliable-client handling and transparent reserve
advice for smaller portfolios. It is positioned as a module that could integrate after validation.

**Is the affordability number proven?**  
No. The unsupported ₹46.50 allocation is withdrawn. Itemised covered costs estimate
about ₹153/home/month at100 payers (base), excluding unquoted retrofit/access and
energy changes. This is cost recovery, not a validated affordable price. Refer to
docs/AFFORDABILITY.md for assumptions, sources, sensitivities and missing agreements.

## What not to claim

The daily LP sees a day's sequence of rolling forecasts retrospectively. Its uncertainty monitor
is causal, but live dispatch needs rolling planning. This is a schedule benchmark.

## Five-minute delivery and demo

Use 30 seconds for purpose, 30 for the operator journey, 40 for architecture, 45 for model evidence,
40 for fault/fallback handling, 35 for scale, 50 for uncertainty/LP, and 30 for affordability and
the prototype-round ask. Keep equations available for questions. The main story is that an operator
gets a forecast, knows when to distrust it, and sees whether limited backup can cover the requirement.

Build the replay with `python build_dashboard.py`, then launch `python -m streamlit run dashboard.py`.
Show the faulty scenario, trust, update reasons, conservative margins and capacity warnings. Open
the evidence tabs for model/scale comparisons. Do not train full experiments during the pitch.
See `GridMesh_Judge_QA.md` for detailed answers.

- Do not call the system field-deployed or utility-certified.
- Do not call model updates private by default.
- Do not claim Schneider lacks federated learning or an equivalent internal feature.
- Do not present synthetic demand or simulated assets as measured feeder data.
- Do not describe the statistical margin target as an end-to-end reliability guarantee.
