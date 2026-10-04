# Proposed pilot and operations

This is an assumed service design, not an existing community, installation, partnership or access agreement.

## One neighbourhood, 100 participating homes

Target: a Bengaluru peri-urban housing cluster, prioritising dependable daytime essential services. Assumed existing aggregated solar100kW, demand80kW, grid availability50kW, generic backup power30kW and scheduled-energy allowance60kWh/day. These are deterministic demo assumptions, NOT the research experiment's asset capacities. Fifty versus100 paying homes is an economics participation sensitivity, not a forecast-client count.

Backup technology remains unspecified. Daily reset assumes verified replenishment by the asset owner before service starts. It is not a battery state-of-charge model. Owner must confirm fuel/charging, efficiency, response time, protection, available power and energy; otherwise no operational pilot. Advisory interval/hourly weather gives +60-minute planning, not sub-minute protection or an actual response-time guarantee.

## Responsibilities and flows

Proposed asset owner: association or local ESCO, to be selected. Proposed access: documented metering and advisory-use permission; electrical dispatch stays with qualified owner personnel. No household is assumed to have consented.

Local operator validates freshness, demand/grid estimate and available assets, reviews suggested backup, then approves or rejects through existing safe procedures. Any uncovered reserve triggers refresh/owner escalation, followed by DISCOM coordination if authorised. If inputs or assets cannot be confirmed, suspend advice for operational use. Do not promise emergency supply or direct households to bypass safety procedures.

Qualified maintainer checks metering, communications and backup health under an agreed service contract. Team BP maintains forecast software, versioned configuration, regression checks and incident records. Association/ESCO handles consent, cost collection and complaints. DISCOM retains grid coordination authority.

Data: local histories -> local training -> shared model updates -> aggregate forecast -> uncertainty and limited-backup advice -> operator. Raw history stays local by design, but no formal privacy/cybersecurity certification. Physical energy: grid/solar/owner backup -> homes, outside software control. Money: consenting paying homes -> local operator -> software and separately agreed suppliers; sponsor/financing needed for upfront costs. These flows are proposals only.

DISCOM handoff, subject to permission: timestamped forecast, uncertainty, expected shortfall, reserve requirement and site-health signals. CSV/manual review first. No fabricated Schneider API or live integration. Schneider already offers grid monitoring, control and DER/microgrid solutions; GridMesh proposes a neighbourhood advisory input/workflow, not replacement or proven superiority. Official capability reference: https://www.se.com/ww/en/work/solutions/electric-utilities/smart-grid/ (checked2026-10-04).

## Four proposed phases

1. Shadow mode: four weeks of measured solar/load/grid and backup availability, advice only. Log issuance time, freshness, recommendation, actual outcome and reason for rejection. No physical dispatch by software.
2. Operator pilot: only after owner permission, safety/maintenance procedures and full cost funding. Keep human approval and fallback/escalation.
3. DISCOM coordination: only after approval and agreed information handoff.
4. Replicate: only after measured benefit, operator usability and affordability are established.

Before prospective evaluation, preregister paired fixed20 and frozen candidate advice on identical intervals/assets. Primary gates: candidate fully supplied daylight intervals no worse than fixed20, ENS no worse, less scheduled backup; proposed availability target99% when feasible. Record unmet target rather than relax it after results. Require no power/energy constraint violation, clear warning for missing input, and operator sign-off on each failure scenario. Night supply/outage-hour claims require a separate scope and study.

## Quote checklist and interviews (not sent)

Current affordability model: `configs/affordability_costs.yaml`, reproduced by
`build_full_service_costs.py`. See `docs/AFFORDABILITY.md` for itemised rates,
paid operator/support assumptions and50/100-household cost recovery. Historical
4650 software allocation is withdrawn. Existing staff time is paid incrementally;
additional owner access and physical retrofit remain unquoted. Association/ESCO
must confirm initial setup cash and recurring shortfall funding before any offer.

Quotes: installed sensor/gateway and safety works; eligible connectivity tariff and tax; qualified labour hours/rate; maintenance and warranty; owner access; backup technology, kW/kWh, replenishment efficiency and duty cycle; replacement life; financing repayment and tax; upfront funder. Record date, location, validity, exclusions and payer. Current public Chennai BSNL promotion is not a Bengaluru installed quote and is not used in costs (official example: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2299697&lang=2&reg=48).

Ask residents: Which daytime interruptions hurt most? Which essential loads matter? Would forecast/risk advice help? Who should operate it? What monthly fee would you accept, including energy/maintenance? Would50 or100 homes participate? Ask owner: Who controls backup and replenishment? What capacity is genuinely available? Ask operator: Is each warning/action understandable? Ask DISCOM: Which approved signals would be useful? No interviews, enquiries or external contacts have been made.
