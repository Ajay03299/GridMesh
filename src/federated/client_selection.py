"""Which clients take part in a round.

1. client sampling  : a random `client_fraction` of all clients (1.0 = everyone)
2. event-aware      : (optional) stable clients join only with prob `stable_participation`;
                      clients with a drift event have priority within the sampling cap
3. dropout          : each selected client silently fails with prob `dropout_rate`
"""
import math


def select_clients(names, rng, fcfg, event_cfg=None, drifting=(), round_idx=1, quarantined=()):
    unavailable = set(fcfg.get("unavailable_sites", []))
    names = [n for n in names if n not in unavailable]
    # Periodic probes let quarantined clients demonstrate recovery without contributing
    # to ordinary rounds. Their trust gate still controls aggregation of a probe update.
    if round_idx % fcfg.get("recovery_probe_every", 3):
        names = [n for n in names if n not in quarantined]
    if not names:
        return [], [], []
    k = max(1, math.ceil(fcfg["client_fraction"] * len(names)))
    cap = fcfg.get("max_clients_per_round")
    if cap:
        k = min(k, int(cap))
    if event_cfg is not None:
        urgent = [n for n in names if n in drifting]
        if len(urgent) > k:
            urgent = list(rng.choice(urgent, k, replace=False))
        others = [n for n in names if n not in urgent]
        sampled = urgent + list(rng.choice(others, k - len(urgent), replace=False))
    else:
        sampled = list(rng.choice(names, size=k, replace=False)) if k < len(names) else list(names)
    skipped = []
    if event_cfg is not None and round_idx > 1:
        keep = []
        for n in sampled:
            if n in drifting or rng.random() < event_cfg["stable_participation"]:
                keep.append(n)
            else:
                skipped.append(n)
        sampled = keep
        if cap and len(sampled) > cap:
            urgent = [n for n in sampled if n in drifting]
            stable = [n for n in sampled if n not in drifting]
            sampled = urgent[:cap] + stable[:max(0, cap - len(urgent))]
    dropped = [n for n in sampled if rng.random() < fcfg["dropout_rate"]]
    active = [n for n in sampled if n not in dropped]
    return active, skipped, dropped
