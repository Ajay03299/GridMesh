"""A federated client = one renewable site. Raw data never leaves this object;
only model parameters and a few summary numbers are returned to the server."""
from dataclasses import dataclass

from src.models.forecasting_model import (build_model, get_params, mse_on, set_params,
                                          train_epochs)
from src.reliability.trust import data_quality


@dataclass
class Update:
    name: str
    params: list          # updated model parameters
    n_samples: int        # local training samples (FedAvg weight)
    e_global: float       # val MSE of the RECEIVED global model on local data (reliability signal)
    e_local: float        # val MSE after local training
    quality: float        # fresh-reading rate of local sensor data


class FLClient:
    def __init__(self, data, cfg, seed):
        self.data, self.cfg, self.seed = data, cfg, seed
        self.name = data.params.name
        self.model = build_model(data.train.X.shape[1], cfg["model"], seed)
        self.faulty_active = False     # switched on by the server at faults.start_round

    @property
    def current(self):
        """The data the site actually has right now (degraded copy while faulty)."""
        return self.data.faulty if (self.faulty_active and self.data.faulty) else self.data

    def evaluate(self, global_params):
        """Heartbeat: score a global model on my recent validation window (no training)."""
        set_params(self.model, global_params)
        d = self.current
        return mse_on(self.model, d.val), data_quality(d.val)

    def fit(self, global_params, round_idx):
        d = self.current
        e_global, quality = self.evaluate(global_params)
        train_epochs(self.model, d.train.X, d.train.y_model, self.cfg["federated"]["local_epochs"],
                     self.cfg["model"], self.seed + 1000 * round_idx + self.data.params.site_id)
        return Update(self.name, get_params(self.model), len(d.train.y),
                      e_global, mse_on(self.model, d.val), quality)
