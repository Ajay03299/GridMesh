"""Two-stage trust aggregation, simulated in one process.

Site parameters first form neighbourhood models. Feeder weights use represented sample
counts, neighbourhood validation errors and quality. Composing these weights is exactly
equivalent to forming the intermediate parameter arrays and averaging them at the feeder.
"""
from types import SimpleNamespace

from src.federated.reliability_aware import ReliabilityAwareFedAvg


class HierarchicalReliabilityFedAvg:
    def __init__(self, rcfg, names, group_size=25):
        self.groups = {name: f"N{i // group_size}" for i, name in enumerate(names)}
        self.local = {g: ReliabilityAwareFedAvg(rcfg) for g in set(self.groups.values())}
        self.feeder = ReliabilityAwareFedAvg(rcfg)

    def weights(self, updates, peer_errors=None):
        within, group_updates, trust, quarantined, reasons = {}, [], {}, [], {}
        for group in sorted(self.local):
            members = [u for u in updates if self.groups[u.name] == group]
            if not members:
                continue
            peers = {n: e for n, e in (peer_errors or {}).items() if self.groups[n] == group}
            weights, info = self.local[group].weights(members, peers)
            within.update(weights)
            trust.update(info["trust"])
            quarantined.extend(info["quarantined"])
            reasons.update(info["quarantine_reasons"])
            represented = sum(u.n_samples for u in members if weights[u.name] > 0)
            if represented:
                group_updates.append(SimpleNamespace(name=group, n_samples=represented,
                    e_global=sum(weights[u.name] * u.e_global for u in members),
                    quality=sum(weights[u.name] * u.quality for u in members)))
        group_weights, group_info = self.feeder.weights(group_updates) if group_updates else ({}, {})
        weights = {u.name: within.get(u.name, 0) * group_weights.get(self.groups[u.name], 0)
                   for u in updates}
        return weights, {"trust": trust, "quarantined": quarantined,
                         "quarantine_reasons": reasons, "group_weights": group_weights,
                         "group_health": group_info, "reporting_groups": len(group_updates),
                         "fallback": "last_trusted_model" if not sum(weights.values()) else None}
