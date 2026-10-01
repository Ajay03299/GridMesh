"""Threshold drift detector for event-aware participation (kept deliberately simple).

drift_detected  <=>  e_now > mean(history) + k * std(history)
                     AND e_now > (1 + min_rel_increase) * mean(history)
(the second condition stops tiny wobbles from counting as drift when errors are very stable)

`history` = the client's last `window` reported errors. Needs `min_history` points first.
While a model is converging, errors fall, so the rule stays quiet; a sudden jump
(sensor fault, weather-regime change) triggers it.
"""
import numpy as np


class DriftDetector:
    def __init__(self, k=2.0, window=5, min_history=3, min_rel_increase=0.0):
        self.k, self.window, self.min_history = k, window, min_history
        self.min_rel = min_rel_increase
        self.history = {}

    def update(self, name, error):
        """Record a new error for `name`; return True if it is a drift event."""
        h = self.history.setdefault(name, [])
        drift = False
        if len(h) >= self.min_history:
            recent = np.array(h[-self.window:])
            drift = (error > recent.mean() + self.k * max(recent.std(), 1e-12)
                     and error > (1 + self.min_rel) * recent.mean())
        h.append(float(error))
        return bool(drift)
