"""Reliability-aware FedAvg (OUR METHOD — a transparent prototype, not a published formula).

    final_weight_k = n_k * trust_k * quality_k        (0 if trust_k < quarantine threshold)
    weights are then normalised to sum to 1.

n_k       : local sample count (same as FedAvg)
trust_k   : smoothed reliability from the error of the global model on k's data
            (see src/reliability/trust.py)
quality_k : fresh-reading rate of k's sensor data (missing / frozen readings lower it)
"""
from src.reliability.trust import instant_reliability, update_trust


class ReliabilityAwareFedAvg:
    name = "reliability_fedavg"

    def __init__(self, rcfg):
        self.rcfg = rcfg
        self.trust = {}

    def weights(self, updates, state=None):
        r_now = instant_reliability({u.name: u.e_global for u in updates},
                                    self.rcfg["error_tolerance"], self.rcfg["error_sharpness"])
        self.trust = update_trust(self.trust, r_now, self.rcfg["trust_ema"])
        raw, quarantined = {}, []
        for u in updates:
            t = self.trust[u.name]
            q = u.quality if self.rcfg["use_data_quality"] else 1.0
            if t < self.rcfg["quarantine_below"]:
                quarantined.append(u.name)
                raw[u.name] = 0.0
            else:
                raw[u.name] = u.n_samples * t * q
        total = sum(raw.values())
        if total == 0:                      # everyone quarantined -> fall back to FedAvg
            total, raw = sum(u.n_samples for u in updates), {u.name: u.n_samples for u in updates}
        info = {"trust": dict(self.trust), "reliability_now": r_now, "quarantined": quarantined}
        return {k: v / total for k, v in raw.items()}, info
