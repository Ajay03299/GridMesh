# GridMesh affordability: itemised planning, not validated pricing

## Decision

Withdraw the unsupported INR4,650/month software allocation and its INR46.50/93
household divisions from active pricing. Use the editable
`configs/affordability_costs.yaml` with `build_full_service_costs.py` instead.
Output: `outputs/affordability_v1/COST_MODEL.md`, JSON and CSV scenarios.

The covered existing-assets planning budget is INR6,996 / 15,268 / 40,874 per month
(low/base/high, rounded). At 100 payers this is about INR70 / 153 / 409 per month;
at 50 payers about INR140 / 305 / 817. These are recovery of COVERED items only,
not all-in electricity/service prices or a validated affordable fee. The bands
combine published infrastructure rates with explicitly assumed paid task hours,
rates, currency conversions and contingencies. They are scenario envelopes,
not statistical confidence intervals or supplier quotes.

## Trace of the withdrawn allocation

The earlier `configs/pilot_costs.yaml` called INR4,650 a "Team submission
hypothesis; hosting/software combined, allocation unverified". Earlier
`configs/full_service_costs.yaml` likewise called it an existing team illustration.
The v34 Slide-11 speaker notes explicitly said "original pre-tax software/hosting
illustration only, not real sourced quote". No itemised breakdown, invoice or
supplier quote supported it in those records.

Older presentation builders repeat the literal: `work/showcase/build_showcase.mjs`,
`build_canva.mjs` and `build_product_pitch.mjs`. One calls it sourced-deck content,
but does not identify any invoice or cost calculation. The local six-page mentor
reference `work/mentor_reference.pdf` contains the forecasting/optimization
requirements, but no4650 allocation or itemised pricing. The original Downloads
PPT path is currently unavailable. The inspected local records do not justify
the allocation. Do not manufacture a breakdown to fit the historic total.

Historical decks, prior packages and frozen outputs remain unchanged for
traceability and must not be used as current prices. Legacy cost configurations
now leave the unsupported active amount null. The new submission supersedes
the v34 economics only, preserving all research results and slide order.

## Smallest proposed deployment and responsibilities

One 100-home Bengaluru community, existing solar and generic backup, advisory
software and owner-operated physical dispatch. This is one feeder service, not
100 physical federated computers. The research's 100 virtual sites are separate.

- Assets: existing association/ESCO owner; original equipment purchase stays
  sunk/existing, not charged again. Owner confirms availability and replenishment.
- GridMesh operation: a paid existing local operator reviews daily advice and
  exception warnings. Incremental time is budgeted, not assumed free.
- Backup: qualified asset-owner personnel arrange safe dispatch and buy energy.
  No software equipment commands or implicit guarantee.
- Service payer: consenting households pay the association; the association
  contracts hosting, connection, paid support and data-health maintenance.
- Existing physical maintenance, current owner access and energy remain paid
  under existing agreements. Amounts are required for total operations, not
  silently set to zero. Any extra availability fee is a separate incremental item.
- Readable meter exports, local training compute and permitted access must exist
  or have a retrofit quote. The model leaves that upfront cost TBD; even the low
  band does not claim that existing equipment is adequate or free to access.
- New-equipment deployment is separate: installed technology-specific capex and
  lifetime/finance quote required, not an addition hidden in the existing case.

All roles and funding arrangements remain proposed, not contracted.

## Why these inputs are defensible planning inputs

Published regular supplier rates checked 2026-10-04:

- [AWS Lightsail](https://aws.amazon.com/lightsail/pricing/): public IPv4 Linux
  4GB USD24/month and 8GB USD44/month; object-storage tiers USD1/3/5 per month.
  Mumbai halves headline transfer allowances. No promotional credits used.
  Capacity, model training and traffic eligibility require a pilot load test.
- [AWS India tax table](https://aws.amazon.com/tax-help/apac-vat-rates/): 18% GST.
- [Airtel Bengaluru](https://www.airtel.in/plans/broadband/bangalore/): ordinary
  INR499/month 40Mbps + GST. Exact address and permitted community use need
  confirmation. No free-installation offer credited.
- [CBIC telecom tax](https://cbic-gst.gov.in/pdf/press-release/GST-telecom-services.pdf):
  18% telecom GST, consistent with the supplier's additional-GST disclosure.

Task assumptions are independent of the old total: operator 15/30/60 minutes per
day at INR150/250/400 per paid hour; support 2/4/8 hours/month at INR500/750/1000;
data-health maintenance 1/2/4 hours/month at the same technical rate. These are
negotiation/planning rates, NOT prevailing wages, observed durations or legal
minimum wages. Operator hourly allowance includes employment overhead by
assumption. No 24-hour callout support included. Technical contractor tax allowance
18% is a scenario assumption pending supplier status.

Onboarding covers 10/20/40 technical hours, paid operator training 2/4/8 hours and
connection commissioning allowance INR1500/2500/4000. It excludes electrical
meter retrofit. Known setup is INR7,970/21,650/55,120 tax inclusive, recovered over
36 months. An association initial-cash funder is still needed; a recovery
provision is not a loan or confirmed sponsor. USD/INR90/95/100 is explicitly a
conversion stress assumption, not a quoted current exchange rate. Contingency
10/15/20% applies to covered recurring costs and setup provision, not an assumed
ceiling on unquoted items. No input-tax credit assumed.

## Base calculation and funding

Monthly recurring cash: INR12,675.52, comprising hosting/archive3,026.70,
connection588.82, operator3,750, support3,540 and data-health maintenance1,770.
Setup provision21,650/36 =601.39/month. Contingency15% =1,991.54/month.
Covered service budget =15,268.45/month, zero commercial margin.

At 100 paying homes the covered cost-recovery contribution is152.6845/home/month.
At 50 it is305.3689. If50 pay only the100-home contribution, the covered deficit
is7,634.22/month before unknown costs. Association reserves or a named sponsor
must fund it, neither confirmed. Contributions should use exact calculations
and agreed rounding; the slide's whole-rupee figures are presentation rounding.

No subsidy, free staff, DISCOM payments or signed funding assumed. The model
accepts a named sponsor but ignores unconfirmed money in ordinary outputs.
An explicitly requested conditional-funding scenario may include it and labels
it conditional. A commercial selling price, sales tax, margin and development
capital recovery need separate terms; this is cost recovery, not a business quote.

## Benefit and feasibility

The primary simulation reliability result remains96.89% to99.39% fully supplied
daytime intervals, with lower realized unserved energy. It does not establish
household outage hours, income benefit or willingness to pay.

Scheduled backup27.04 to16.76MWh is not consumed backup. Operator demo CSVs contain
an idealised `actual_delivered_backup_mw = min(schedule, actual gap)` field, but
these are five synthetic intervals without real asset efficiency, fuel, charging,
wear or pilot-month dispatch. They cannot price annual household savings. No
monetary benefit credited. Abstract cost scores never become rupees.

Full incremental cost needs meter/edge retrofit, additional owner access and the
signed change in actual consumed energy/asset wear. Total operations additionally
needs the post-deployment actual backup-energy bill, existing owner access and
physical O&M. Outputs remain TBD, not zero or silently complete.

Conclusion: a specific paid service budget can be examined, but affordability
and complete economic feasibility remain unresolved. At low participation the
cost burden doubles. Prefer a shadow pilot run by an existing operator, with
quotes and paid-time logs, before a household offer. Sharing central support
across communities is a hypothesis to cost and load-test; do not divide all
local costs across hypothetical homes. An existing operator may fund a pilot
only after an explicit agreement, never as an assumed subsidy.

See the generated cost report for sensitivities, equations and the short quote/
interview checklist. No enquiries, purchases or external contacts were made.
