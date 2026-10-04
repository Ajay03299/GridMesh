# Round1 delivery and remaining evidence

Affordability correction: use the new affordability submission with unchanged
research slides and itemised Slide11. Refer to `docs/AFFORDABILITY.md` and
`outputs/affordability_v1`, not the older4650 allocation outputs. The verification
record below describes v34; affordability verification is in the new package.

## Submission files

- Editable12-slide deck: `GridMesh_Professional_Submission_v34.pptx`.
- Matching12-page PDF: `GridMesh_Professional_Submission_v34.pdf`.
- Overview: `docs/ROUND1_OVERVIEW.md`,444words including heading, within300–500.

## Required deliverables

| Requirement | Delivered | Boundary |
|---|---|---|
| Solution/problem/assumptions/India suitability | Deck1–3,6–8,12 and findings | One-city weather, modeled PV and virtual co-located sites. |
| Architecture and data/energy flows | Deck3–4 | Digital advisory, not physical dispatch. Pilot document adds proposed money flow. |
| UX/process/data/model artifacts | Deck5–6,10, operator scenarios and configurations | Synthetic demo and virtual clients, not installations. |
| Prototype or simulation | Reused Streamlit dashboard,5operator cases | No hardware build or live utility integration. |
| Reliability versus baseline | Frozen five-seed table on Slide7 | Daytime supplied intervals and ENS, not measured outage hours. |
| Ownership/O&M/economics | Slide11, pilot document and editable YAML | Proposed responsibilities; quotes, funding and full affordability unresolved. |
| Team introduction | Slide12 | Team BP,4members and both mentors visible. |
| Implementation/innovation/roadmap | Slides3–12 and reports | No guaranteed grade or competition outcome. |

## Verified locally

22unit regression tests passed; audit36/36; operator view15paths; dashboard8dark/light replay cases plus constrained/stale/exhausted advice; frozen operational replay70test/270validation rows matched. New3-seed fault training18runs and5-seed operating160test/280validation rows completed. Original evidence hashes unchanged. Exactly12slides/pages; native evidence tables, charts, theme and Slides1–4,6–8 unchanged. Every exported PDF page inspected. Overview444words.

## What improved, without overstating resolution

Implemented missing/stale/degraded-input safety bounds, mandatory warnings, explicit power/energy/shortfall explanations and reproducible scenarios. Experimental joint net-error margin selected on validation only, not promoted as universal default. Current weighting claim narrowed after mixed results. Full-service model includes upfront/monthly categories, payers, sources and taxes; missing costs remainTBD. Proposed pilot includes permissions, replenishment, qualified maintenance and escalation.

## External checks not completed

Measured site data/telemetry latency, prospective shadow trial, actual backup/access/safety agreement, DISCOM permission, installed supplier/tariff/tax quotes, upfront sponsor, recurring funding and resident willingness to pay. No real community deployment, partnership, formal privacy, field uptime or proven all-in price.

The original repository was already dirty. Relevant changes remain local and uncommitted.
No push, deployment, enquiry, hardware purchase or external contact was performed.
