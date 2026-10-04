# Supported claims and boundaries

| Claim | Status | Evidence / boundary |
|---|---|---|
| Frozen primary nominal policy improves daytime supply and uses less scheduled backup than fixed20 | Supported in simulation | Five seeds:96.89->99.39%, ENS0.875->0.128MWh, backup27.04->16.76MWh. Modeled PV/synthetic demand/known grid and load. |
| FL exceeds local/persistence forecast baselines | Conditional | Chronological co-located virtual-site dataset; training budgets differ. No universal or geographic claim. |
| Weighting consistently improves current faulty-data performance | Unsupported | Three paired seeds show mixed results. Experimental option. Legacy8.9% belongs to separate4-site test. |
| Selective uploads reduce physical network traffic45.55% | Unsupported wording | Supported only as modeled protocol bytes; +2.72% relative forecast-error trade-off. |
| Reliability guard meets99% in all operating conditions | Unsupported | Validation target fails under several stresses; fixed fallback is not a guarantee. |
| Stale/missing solar invokes explicit safety bound and warning | Implemented/tested | It is zero-solar planning, not a trusted prediction. Missing demand/assets remain rejected. |
| End-to-end operator advice and five scenarios exist | Implemented/tested | Synthetic demo, proposed approval action; no equipment dispatch. |
|100sites are100household installations | Unsupported | Research clients are virtual sites in one process. Demo100homes is an independent assumed pilot. |
| INR46.50 is an affordable all-in fee | Withdrawn | Unsupported historical4650 allocation removed from active pricing; no itemised justification. |
| Itemised covered service budget is inspectable | Implemented | Published supplier inputs + explicit task assumptions, low/base/high,50/100 participation, taxes/setup/contingency. Full cost and WTP remain unresolved. |
| Schneider complementarity | Proposed | Official capability source supports their existing grid/DER offerings; no claimed partnership/API/superiority. |
| Formal privacy, field uptime, outage hours, bill/emissions savings | Unsupported | None measured or proven. |

Experiments remain separate: legacy4sites/+30min; frozen blocked-month100-site Bengaluru model comparison; current chronological100-site/+60min; new3-seed faults; new5-seed operating stresses. Descriptive means are not confidence intervals.
