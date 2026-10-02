"""Interchangeable MLP, GRU and LSTM forecasters used by every training method."""
import copy

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPForecaster(nn.Module):
    def __init__(self, n_features, hidden=(64, 64)):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class RecurrentForecaster(nn.Module):
    """Encode lagged variables as a sequence, then add issue-time/static features."""
    def __init__(self, n_features, hidden, lookback, sequence_features, cell="gru"):
        super().__init__()
        self.lookback = int(lookback)
        self.sequence_features = int(sequence_features)
        self.sequence_width = self.lookback * self.sequence_features
        if self.sequence_width > n_features:
            raise ValueError("lookback * sequence_features exceeds the feature vector width")
        rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
        recurrent_hidden = int(hidden[0])
        self.rnn = rnn_cls(self.sequence_features, recurrent_hidden, batch_first=True)
        static_width = n_features - self.sequence_width
        head_hidden = int(hidden[1] if len(hidden) > 1 else recurrent_hidden)
        self.head = nn.Sequential(
            nn.Linear(recurrent_hidden + static_width, head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, 1))

    def forward(self, x):
        # Feature construction is variable-major: [v1_lag0..L, v2_lag0..L, ...].
        seq = x[:, :self.sequence_width].reshape(
            -1, self.sequence_features, self.lookback).transpose(1, 2)
        out = self.rnn(seq)
        hidden = out[1][0] if isinstance(out[1], tuple) else out[1]
        encoded = hidden[-1]
        static = x[:, self.sequence_width:]
        return self.head(torch.cat([encoded, static], dim=1)).squeeze(-1)


def build_model(n_features, mcfg, seed):
    # Small CPU models are slower when each minibatch spawns many BLAS workers.
    torch.set_num_threads(mcfg.get("cpu_threads", 1))
    torch.manual_seed(seed)
    architecture = mcfg.get("architecture", "mlp").lower()
    hidden = tuple(mcfg["hidden"])
    if architecture == "mlp":
        return MLPForecaster(n_features, hidden)
    if architecture in ("gru", "lstm"):
        return RecurrentForecaster(
            n_features, hidden, mcfg["lookback"], mcfg["sequence_features"], architecture)
    raise ValueError(f"Unknown model architecture '{architecture}'")


# ---- parameter exchange (what federated clients send / receive) ----
def get_params(model):
    return [p.detach().cpu().numpy().copy() for p in model.state_dict().values()]


def set_params(model, params):
    keys = list(model.state_dict().keys())
    model.load_state_dict({k: torch.tensor(v) for k, v in zip(keys, params)})


def n_bytes(params):
    return int(sum(p.nbytes for p in params))


# ---- training / inference ----
def train_epochs(model, X, y, epochs, mcfg, seed):
    """Plain mini-batch Adam on MSE. Returns mean training loss of the last epoch."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=mcfg["lr"], weight_decay=mcfg["weight_decay"])
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    bs, loss_fn = mcfg["batch_size"], nn.MSELoss()
    model.train()
    last = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(yt), generator=g)
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        last = total / len(yt)
    return last


@torch.no_grad()
def predict(model, split):
    """Final forecast in p.u. = baseline + learned correction, clipped to [0, 1]."""
    model.eval()
    raw = split.base + model(torch.from_numpy(split.X)).numpy()
    # Preserve invalidity for the validation/fallback layer; clipping infinity to 1
    # would otherwise hide a failed model behind a plausible physical bound.
    return np.where(np.isfinite(raw), np.clip(raw, 0.0, 1.0), np.nan)


def mse_on(model, split):
    """Validation loss used for model selection (daytime targets only)."""
    pred = predict(model, split)
    m = split.daytime
    return float(np.mean((pred[m] - split.y[m]) ** 2))


def fit_with_best_val(model, train, val, mcfg, seed, epochs=None, frac=1.0):
    """Train for `epochs`, keep the checkpoint with the lowest validation MSE.
    frac < 1: each epoch trains on a random share of the rows (pooled benchmark only)."""
    epochs = epochs or mcfg["epochs"]
    best, best_state, history = np.inf, None, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        X, y = train.X, train.y_model
        if frac < 1.0:
            take = rng.choice(len(y), max(1, int(len(y) * frac)), replace=False)
            X, y = X[take], y[take]
        tr = train_epochs(model, X, y, 1, mcfg, seed + ep)
        v = mse_on(model, val)
        history.append({"epoch": ep + 1, "train_mse": tr, "val_mse": v})
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history


if __name__ == "__main__":
    # Self-check:  python -m src.models.forecasting_model   (trains site A alone)
    import time
    from src.data.adapter import load_config
    from src.data.preprocessing import prepare_clients
    from src.evaluation.metrics import (regression_metrics, persistence_forecast,
                                        smart_persistence_forecast, skill_score)

    cfg = load_config()
    set_seed(cfg["seed"])
    clients, _ = prepare_clients(cfg, n_clients=4, seed=cfg["seed"])
    c = clients[0]
    model = build_model(c.train.X.shape[1], cfg["model"], cfg["seed"])
    print(f"Site {c.params.name}: {c.n_train} train samples, "
          f"{sum(p.numel() for p in model.parameters())} model parameters "
          f"({n_bytes(get_params(model)) / 1024:.1f} KB per update)")
    t = time.time()
    model, hist = fit_with_best_val(model, c.train, c.val, cfg["model"], cfg["seed"])
    for h in hist[::3] + [hist[-1]]:
        print(f"  epoch {h['epoch']:>2}  train_mse {h['train_mse']:.5f}  val_mse {h['val_mse']:.5f}")
    print(f"Trained in {time.time() - t:.1f}s")

    ts, day = c.test, c.test.daytime
    rows = {"MLP (site A only)": predict(model, ts),
            "Persistence": persistence_forecast(ts),
            "Smart persistence": smart_persistence_forecast(ts)}
    ref = regression_metrics(ts.y, rows["Smart persistence"], day)["rmse"]
    print(f"\nTEST, daytime only, p.u. of capacity (n={day.sum()}):")
    for name, pred in rows.items():
        m = regression_metrics(ts.y, pred, day)
        print(f"  {name:<20} MAE {m['mae']:.4f}  RMSE {m['rmse']:.4f}  bias {m['bias']:+.4f}  "
              f"skill vs smart-pers {skill_score(m['rmse'], ref):+.3f}")
