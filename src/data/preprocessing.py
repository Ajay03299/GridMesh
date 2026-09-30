"""Feature engineering, supervised windows and the chronological split.

Leakage rules:
  * every sample is assigned to train/val/test by the time of its TARGET (t + horizon);
  * inputs only use values at or before t, except `future_known` covariates that are
    deterministic in advance (clear-sky irradiance, sun angle, calendar);
  * each site standardises features with statistics from ITS OWN training split only
    (a site never sees another site's data — this holds for every method, so the
    comparison between methods is fair).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.adapter import load_dataset
from src.data.virtual_sites import apply_fault, generate_virtual_sites
from src.evaluation.metrics import smart_persistence


@dataclass
class Split:
    X: np.ndarray            # float32 features (standardised)
    y: np.ndarray            # float32 target, p.u. at t+h
    daytime: np.ndarray      # bool, target is during daylight (metrics use these)
    pv_now: np.ndarray       # p.u. at t (persistence baseline)
    cs_now: np.ndarray       # clear-sky GHI at t
    cs_future: np.ndarray    # clear-sky GHI at t+h
    time: np.ndarray         # timestamp of the target
    base: np.ndarray         # baseline forecast the model corrects (smart persistence or 0)
    stale: np.ndarray        # bool, raw GHI reading identical to previous step (missing/frozen)

    @property
    def y_model(self):
        """What the network is trained on: the correction to the baseline."""
        return (self.y - self.base).astype(np.float32)


@dataclass
class ClientData:
    params: object           # SiteParams
    train: Split
    val: Split
    test: Split
    feature_names: list
    data_quality: float      # 1 - fraction of missing sensor readings
    faulty: object = None    # degraded copy of this site's data (fault experiments only)

    @property
    def n_train(self):
        return len(self.train.y)


def add_derived_features(df, fcfg):
    df = df.copy()
    cs = df["Clearsky GHI"].astype(float)
    df["kt"] = np.where(cs > 20, df["GHI"] / cs.clip(lower=1), 0.0).clip(0, 1.5)
    df["cos_zenith"] = np.cos(np.radians(df["Solar Zenith Angle"]))
    ts = df["timestamp"]
    hour = ts.dt.hour + ts.dt.minute / 60.0
    doy = ts.dt.dayofyear
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["doy_sin"], df["doy_cos"] = np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365)
    for v in fcfg["cloud_type_values"]:
        df[f"cloud_{v}"] = (df["Cloud Type"] == v).astype(float)
    return df


def make_windows(df, fcfg):
    """Build the feature matrix. Row i = forecast issued at time t_i for t_i + horizon."""
    L, h = fcfg["lookback"], fcfg["horizon"]
    cols = {}
    for c in fcfg["lagged"]:
        for lag in range(L):
            cols[f"{c}_lag{lag}"] = df[c].shift(lag)
    for c in fcfg["current"]:
        cols[c] = df[c]
    for v in fcfg["cloud_type_values"]:
        cols[f"cloud_{v}"] = df[f"cloud_{v}"]
    for c in fcfg["future_known"]:
        cols[f"{c}_t+h"] = df[c].shift(-h)
    X = pd.DataFrame(cols)

    aux = pd.DataFrame({
        "y": df["pv"].shift(-h),
        "daytime": df["Solar Zenith Angle"].shift(-h) < fcfg["daytime_zenith_max"],
        "pv_now": df["pv"],
        "cs_now": df["Clearsky GHI"],
        "cs_future": df["Clearsky GHI"].shift(-h),
        "time": df["timestamp"].shift(-h),
        "seg_in": df["seg"].shift(L - 1),   # segment of the OLDEST input
        "seg_out": df["seg"].shift(-h),     # segment of the TARGET
        "stale": df["GHI"].diff() == 0,     # on RAW values, before any scaling
    })
    # a window may not straddle two segments (no leakage across train/val/test blocks)
    valid = X.notna().all(axis=1) & aux["y"].notna() & (aux["seg_in"] == aux["seg_out"])
    return X[valid], aux[valid]


TRAIN, VAL, TEST = 0, 1, 2


def segment_labels(timestamps, split_cfg):
    """Label every timestep 0=train, 1=val, 2=test (time-ordered, never random).

    chronological   : position in the whole record (one global cut).
    blocked_monthly : position of the DAY inside its calendar month, so each month
                      contributes its first days to train, then val, then test.
    """
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
    tr, va = split_cfg["train"], split_cfg["val"]
    if split_cfg.get("mode", "chronological") == "chronological":
        frac = np.arange(len(ts)) / len(ts)
    else:
        frac = ((ts.dt.day - 1) / ts.dt.days_in_month).to_numpy()
    return np.where(frac < tr, TRAIN, np.where(frac < tr + va, VAL, TEST))


class Standardiser:
    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-6] = 1.0
        return self

    def transform(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


def _to_split(X, aux, mask, scaler, target_mode):
    a = aux[mask]
    pv_now = a["pv_now"].to_numpy(np.float32)
    cs_now = a["cs_now"].to_numpy(np.float32)
    cs_future = a["cs_future"].to_numpy(np.float32)
    if target_mode == "residual_smart_persistence":
        base = smart_persistence(pv_now, cs_now, cs_future)
    else:  # "direct"
        base = np.zeros(len(a), np.float32)
    return Split(
        X=scaler.transform(X[mask].to_numpy(dtype=np.float64)),
        y=a["y"].to_numpy(np.float32),
        daytime=a["daytime"].to_numpy(bool),
        pv_now=pv_now, cs_now=cs_now, cs_future=cs_future,
        time=a["time"].to_numpy(),
        base=base,
        stale=a["stale"].to_numpy(bool),
    )


def build_client(p, sdf, cfg, scaler=None):
    """Windows + split + scaling for one site. Pass `scaler` to reuse the clean site's
    scaler (a degraded sensor does not re-calibrate the site's preprocessing)."""
    fcfg, tm = cfg["features"], cfg["model"]["target"]
    X, aux = make_windows(add_derived_features(sdf, fcfg), fcfg)
    t, s_out = aux["time"], aux["seg_out"]
    # sample-volume heterogeneity: a site only owns its most recent `history_fraction`
    # of training samples (a younger plant). Val/test stay identical for fair comparison.
    t_site_start = t[s_out == TRAIN].quantile(1 - p.history_fraction)
    m_train = (s_out == TRAIN) & (t >= t_site_start)
    m_val, m_test = s_out == VAL, s_out == TEST
    if scaler is None:
        scaler = Standardiser().fit(X[m_train].to_numpy(dtype=np.float64))
    client = ClientData(
        params=p,
        train=_to_split(X, aux, m_train, scaler, tm),
        val=_to_split(X, aux, m_val, scaler, tm),
        test=_to_split(X, aux, m_test, scaler, tm),
        feature_names=list(X.columns),
        data_quality=1.0 - sdf.attrs["missing_frac"],
    )
    return client, scaler


def prepare_clients(cfg, n_clients=4, seed=42, faulty_client=None, fault_type=None):
    """Full data pipeline: load -> virtual sites -> features -> time-ordered split.
    If `faulty_client` (site index) is given, that site also gets a degraded copy `.faulty`."""
    base_df, meta = load_dataset(cfg)
    seg = segment_labels(base_df["timestamp"], cfg["split"])
    sites = generate_virtual_sites(base_df, cfg, n_clients=n_clients, seed=seed)

    clients = []
    for p, sdf in sites:
        sdf["seg"] = seg
        client, scaler = build_client(p, sdf, cfg)
        if faulty_client is not None and p.site_id == faulty_client:
            bad = apply_fault(sdf, fault_type, cfg["faults"], seed, p.site_id)
            client.faulty, _ = build_client(p, bad, cfg, scaler=scaler)
        clients.append(client)
    ts = pd.Series(base_df["timestamp"])
    info = {"mode": cfg["split"].get("mode", "chronological")}
    for name, k in (("train", TRAIN), ("val", VAL), ("test", TEST)):
        part = ts[seg == k]
        info[name] = f"{part.min():%Y-%m-%d} .. {part.max():%Y-%m-%d} ({part.dt.date.nunique()} days)"
    return clients, info


if __name__ == "__main__":
    # Self-check:  python -m src.data.preprocessing
    from src.data.adapter import load_config
    from src.data.virtual_sites import params_table

    cfg = load_config()
    clients, info = prepare_clients(cfg, n_clients=4, seed=cfg["seed"])
    print(f"Split mode: {info['mode']}")
    for k in ("train", "val", "test"):
        print(f"  {k:<5} {info[k]}")
    print("\nSIMULATED site parameters:")
    print(params_table([(c.params, None) for c in clients]).round(4).to_string())
    print(f"\nFeatures per sample: {len(clients[0].feature_names)}")
    print(f"{'site':<5}{'train':>8}{'val':>8}{'test':>8}{'quality':>9}{'mean pv(day)':>14}")
    for c in clients:
        print(f"{c.params.name:<5}{len(c.train.y):>8}{len(c.val.y):>8}{len(c.test.y):>8}"
              f"{c.data_quality:>9.3f}{c.train.y[c.train.daytime].mean():>14.3f}")
    # leakage guard: no target timestamp is shared between splits
    for c in clients:
        tr, va, te = (set(pd.to_datetime(x.time)) for x in (c.train, c.val, c.test))
        assert not (tr & va) and not (tr & te) and not (va & te)
    months = sorted(pd.to_datetime(clients[0].test.time).month.unique())
    print(f"\nOK: no train/val/test overlap; no window crosses a split boundary.")
    print(f"Test covers months: {months}")
