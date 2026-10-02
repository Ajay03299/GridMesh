# GridMesh reserve optimization

GridMesh separates two decisions that are often mixed together:

1. **How much forecast uncertainty should be covered?**
2. **How should limited backup resources be scheduled?**

That separation keeps the implementation faithful to Khaing, Kannan & Rao (2026) while making the
operational step auditable and small enough for a feeder-level hackathon prototype.

## 1. Causal uncertainty margin

For renewable forecast error \(e_t = y_t-\hat y_t\), GridMesh estimates a rolling mean
\(\mu_{e,t}\) and standard deviation \(\sigma_{e,t}\) using only errors that would already be known
when the new forecast is issued. With target shortfall probability \(\delta\):

\[
m_t = \max\{\Phi^{-1}(1-\delta)\sigma_{e,t}-\mu_{e,t},0\}.
\]

This is an **uncertainty margin**, not the complete backup schedule. The Gaussian rule is inspired
by Theorem 4.3 of the paper. Real weather errors need not be Gaussian, so GridMesh reports realized
availability instead of promising the nominal target. An empirical rolling tail-quantile policy is
implemented as a second, distribution-free benchmark.

## 2. Expected demand gap

Let \(D_t\) be demand, \(\hat R_t\) the aggregate renewable forecast, and \(G_t\) imported grid
power. Grid import is capped at \(\bar G\):

\[
G_t=\min\{\bar G,\max(D_t-\hat R_t,0)\}, \qquad
g_t=\max(D_t-G_t-\hat R_t,0).
\]

The backup requirement is therefore \(b_t=g_t+m_t\). This avoids the earlier conceptual error of
sizing all reserve only from forecast error while ignoring demand.

## 3. Constrained day-ahead schedule

For each day, the scheduler chooses committed backup \(q_t\) and a transparent planned gap \(u_t\):

\[
\min_{q,u}\sum_t \Delta t(c_q q_t+c_{u,t} u_t)
\]

subject to

\[
q_t+u_t\ge b_t,\quad 0\le q_t\le\bar P,\quad
\sum_t q_t\Delta t\le\bar E,\quad u_t\ge0.
\]

Here \(\bar P\) is backup power capacity and \(\bar E\) is the daily energy budget. When energy is
scarce, \(c_{u,t}\) increases with the normalized requirement so the LP serves the highest-need
intervals first; the priority strength is configurable. The planned-gap variable keeps infeasible
operating conditions visible rather than silently clipping them. The prototype solves this linear
program with SciPy HiGHS.

## 4. Evaluation boundary and assumptions

- The prototype evaluates **daytime renewable-intermittency windows**. Night-time resource adequacy
  is explicitly outside this first-round scope.
- Weather and irradiance are real; PV output, sites, demand, faults, capacities, and costs are
  simulated or assumed unless a measured multi-site data file is supplied.
- Grid-import, backup-power, and daily-energy limits are configurable simulation assumptions.
- Scheduled backup is evaluated against realized net deficit. Outputs include backup energy,
  energy not served, availability, planned capacity gap, and total assumed operating cost.
- No result is a utility operating guarantee. A pilot must replace synthetic demand and assumed
  asset limits with DISCOM/feeder measurements and operating constraints.

## 5. Why this is useful to a judge or operator

Every recommendation can be decomposed into: expected gap, uncertainty margin, grid import,
scheduled backup, capacity warning, and realized shortfall. A judge can see what is implemented;
an operator can see why a recommendation was made; and a future pilot can replace assumptions
without changing the full forecasting pipeline.
