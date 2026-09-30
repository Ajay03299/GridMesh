"""Which clients take part in a round.

1. client sampling  : a random `client_fraction` of all clients (1.0 = everyone)
2. event-aware      : (optional) stable clients join only with prob `stable_participation`;
                      clients with a drift event in their last heartbeat ALWAYS join
3. dropout          : each selected client silently fails with prob `dropout_rate`
"""
import math


def select_clients(names, rng, fcfg, event_cfg=None, drifting=(), round_idx=1):
    k = max(1, math.ceil(fcfg["client_fraction"] * len(names)))
    sampled = list(rng.choice(names, size=k, replace=False)) if k < len(names) else list(names)
    skipped = []
    if event_cfg is not None and round_idx > 1:
        keep = []
        for n in sampled:
            if n in drifting or rng.random() < event_cfg["stable_participation"]:
                keep.append(n)
            else:
                skipped.append(n)
        sampled = keep + [n for n in drifting if n not in keep]   # drift overrides sampling
    dropped = [n for n in sampled if rng.random() < fcfg["dropout_rate"]]
    active = [n for n in sampled if n not in dropped]
    return active, skipped, dropped
