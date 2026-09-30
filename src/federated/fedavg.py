"""Vanilla FedAvg (McMahan et al., 2017): weight = local sample count."""


def weighted_average(param_lists, weights):
    total = sum(weights)
    return [sum(w / total * p[i] for p, w in zip(param_lists, weights))
            for i in range(len(param_lists[0]))]


class FedAvg:
    name = "fedavg"

    def weights(self, updates, state=None):
        """Return {client: normalised weight}. `state` unused (kept for a common interface)."""
        total = sum(u.n_samples for u in updates)
        return {u.name: u.n_samples / total for u in updates}, {}
