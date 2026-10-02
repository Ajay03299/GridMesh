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

## The three numbers to remember

At the 5% n-sigma setting, compared with a fixed 20%-of-demand uncertainty margin:

- **34.9% less scheduled backup**
- **10.3% less unserved energy**
- **17.2% lower assumed total cost**

These are five-seed simulation means. Weather and irradiance are real; PV output is modelled;
demand, sites, faults, capacities, limits and costs are simulated or assumed.

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
All three are implemented. In the current common-protocol screen, the compact MLP gave lower error,
less communication and faster local training. We will repeat selection on measured pilot data.

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
No. ₹46.50 per household per month is an illustrative software-service budget at 100 households. It
excludes hardware and energy. A pilot must validate costs, benefit and willingness to pay.

## What not to claim

- Do not call the system field-deployed or utility-certified.
- Do not call model updates private by default.
- Do not claim Schneider lacks federated learning or an equivalent internal feature.
- Do not present synthetic demand or simulated assets as measured feeder data.
- Do not describe the statistical margin target as an end-to-end reliability guarantee.
