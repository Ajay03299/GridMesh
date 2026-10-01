"""Vanilla FedAvg (McMahan et al., 2017): weight = local sample count."""


def weighted_average(param_lists, weights):
    total = sum(weights)
    return [sum(w / total * p[i] for p, w in zip(param_lists, weights))
            for i in range(len(param_lists[0]))]


class FedAvg:
    name = "fedavg"

    def weights(self, updates, peer_errors=None):
        """Return ({client: normalised weight}, info). `peer_errors` unused (common interface)."""
        total = sum(u.n_samples for u in updates)
        return {u.name: u.n_samples / total for u in updates}, {}
