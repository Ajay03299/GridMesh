# GridMesh: judge questions and team answers

**Why XGBoost?** Our weather/PV features are tabular. Boosted trees provide a strong CPU baseline
and readable gain importance. They predict the correction to smart persistence. A depth/leaf
constrained single tree provides a transparent comparison. Gain importance does not prove causality.

**Can you federate those trees with FedAvg?** No. Split structures differ, so tree parameter
averaging would be incorrect. Our XGBoost models are local or pooled. Tree-specific bagging/cyclic
federation remains future work. The collaborative implementation averages compatible neural weights.

**Why FL if local XGBoost is strong?** Local XGBoost is a serious practical candidate. FL addresses
cross-owner collaboration and uneven local histories. Our evidence tests the value of sharing
learning. We do not claim federation must win at every site or under every fault.

**What is novel?** The integrated prototype combines update trust, bounded/event-driven sampling,
rollback/fallback, causal margin health and resource-limited reserve advice. These algorithms are
established. Our contribution is the evaluated integration and community use-case hypothesis.

**How does the reserve follow the research?** Use `e=actual-forecast` and
`m=max(0,Phi^-1(1-delta)*sigma-mu)`, motivated by Khaing/Kannan/Rao Theorem 4.3/Eq.62.
It covers the event `e < -m`. We estimate aggregate residuals only when observable and retain
shared-weather correlation. Our single-node LP is an adaptation rather than full OPF.

**Does 95% mean 95% uptime?** It is a nominal forecast-error margin target under a Gaussian
assumption. Grid/backup shortages and drift can still leave demand unserved. We report actual
margin coverage, interval availability and capacity gaps separately.

**Does a larger reserve always help?** With a finite energy budget, more reserve in one interval
can leave less elsewhere. Nominal, empirical and guarded policies have separate result rows.
This is why the operational evaluation matters in addition to RMSE.

**Can this scheduler dispatch a real feeder?** It currently benchmarks a full day's sequence of
rolling forecasts retrospectively. Margins are causal, but the daily allocation sees that sequence.
Live use requires rolling optimization or actual day-ahead forecasts, operating permissions and
protection/control integration. Operator approval is part of the proposed workflow.

**What happens when sites fail?** Too few accepted updates retain a trusted checkpoint. Runtime
forecasts use available current/last/local output or smart persistence, with source/age/reason
and alerts. If all fallback data are invalid, the system fails explicitly. A whole neighbourhood's
communication loss stays in the evaluation denominator. Local assets still determine service.

**How does quarantine recover?** Every third round permits probes. Healthy relative validation
reports restore EMA trust. All-quarantined rounds do not silently restore bad updates.

**Does 500 clients mean 500 real sites?** Above 20 clients, four compact reference datasets repeat
under new identities. This measures protocol/traffic/runtime behavior. It does not establish
geographic forecasting validity or network availability. Hierarchy reduces feeder-boundary traffic
while adding a coordination hop to total traffic.

**Is FL formally private?** Histories remain local by protocol convention in a single-process
demo. Updates can leak information. Secure aggregation, differential privacy, authentication,
encrypted transport and a privacy assessment remain pilot requirements.

**How does this differ from Schneider?** Official pages already describe forecasting, DER
optimization and microgrid control. GridMesh focuses on cross-owner learning and update reliability,
plus transparent community reserve evidence. Public pages cannot establish Schneider's internal
algorithms. Integration is a future possibility, without an existing partnership claim.

**Who operates and pays?** A community operator/ESCO could own support and escalate to approved
asset operators and the DISCOM. An illustrative INR4,650 monthly software budget is INR46.50 for
100 households or INR93 for 50. Hardware, gateways, energy, tax and financing are excluded. Quotes,
willingness to pay and avoided-cost evidence are required before affordability is established.

**Why are simulated capacities 5–20 MW?** Defaults represent plant portfolios. They are independent
of the 100-household service-cost example. A neighbourhood pilot must supply measured kW-scale
capacities, load and reserve limits. The model's linear scaling does not establish physical or
economic transferability.

**Is the dataset Indian?** Supplied metadata do not verify its year or location. The columns match
an NSRDB-style weather file and 2019 is assumed. Measured Indian multi-site PV/load is a pilot
requirement, rather than a claimed current property.

**What do five seeds prove?** Repeatability under simulated site/sensor/training randomness for
one shared weather record. They are not five independent climates. We show means, variation,
paired fault comparisons and failed targets without tuning on the test set.

**What comes next?** A measured Indian shadow pilot, validation selection before untouched testing,
real asset/cost limits, prospective rolling planning, trustworthy transport and utility-approved
operating procedures. Curtailment and emissions effects remain future measurements.
