"""Synchronous federated training loop (simulation of one server + N sites).

Per round:
  1. select clients (sampling, event-aware skipping, random dropout)
  2. active clients receive the global model, report e_global + quality, train locally
  3. skipping clients send a tiny heartbeat (error of their last received model) if event-aware
  4. server aggregates (FedAvg or reliability-aware) -> new global model
  5. log weights, trust, drift, participation, bytes, per-site validation error
"""
import numpy as np
import pandas as pd

from src.evaluation.experiments import Result
from src.federated.client import FLClient
from src.federated.client_selection import select_clients
from src.federated.fedavg import FedAvg, weighted_average
from src.federated.reliability_aware import ReliabilityAwareFedAvg
from src.models.forecasting_model import build_model, get_params, mse_on, n_bytes, predict, set_params
from src.reliability.drift import DriftDetector

METADATA_BYTES = 32   # n_samples, e_global, e_local, quality sent with each update


def run_federated(clients_data, cfg, seed, method="fedavg", event_aware=False,
                  faulty_client=None, verbose=True):
    fcfg, ecfg = cfg["federated"], cfg["event_aware"]
    rng = np.random.default_rng([seed, 7])
    clients = {c.params.name: FLClient(c, cfg, seed) for c in clients_data}
    names = list(clients)
    faulty_name = clients_data[faulty_client].params.name if faulty_client is not None else None

    global_model = build_model(clients_data[0].train.X.shape[1], cfg["model"], seed)
    gp = get_params(global_model)
    model_bytes = n_bytes(gp)
    last_seen = {n: gp for n in names}
    agg = FedAvg() if method == "fedavg" else ReliabilityAwareFedAvg(cfg["reliability"])
    detector = DriftDetector(ecfg["drift_k"], ecfg["window"], ecfg["min_history"])
    drifting, rows, rounds = set(), [], []

    for r in range(1, fcfg["rounds"] + 1):
        if faulty_name and r >= cfg["faults"]["start_round"]:
            clients[faulty_name].faulty_active = True

        active, skipped, dropped = select_clients(names, rng, fcfg, ecfg if event_aware else None,
                                                  drifting, r)
        updates = [clients[n].fit(gp, r) for n in active]
        for n in active:
            last_seen[n] = gp

        heartbeats = {}
        if event_aware:   # everyone who did not train and did not drop out reports its error
            for n in names:
                if n not in active and n not in dropped:
                    heartbeats[n] = clients[n].evaluate(last_seen[n])[0]
        errors = {u.name: u.e_global for u in updates} | heartbeats
        new_drift = {n for n, e in errors.items() if detector.update(n, e)} if event_aware else set()
        drifting = new_drift

        aggregated = len(updates) >= fcfg["min_clients"]
        weights, info = ({}, {})
        if aggregated:
            weights, info = agg.weights(updates)
            gp = weighted_average([u.params for u in updates], [weights[u.name] for u in updates])
        set_params(global_model, gp)

        # server-side bookkeeping for plots: global model on every site's CLEAN validation data
        val = {n: mse_on(global_model, clients[n].data.val) for n in names}
        comm = len(active) * (2 * model_bytes + METADATA_BYTES) + len(heartbeats) * ecfg["heartbeat_bytes"]
        by_name = {u.name: u for u in updates}
        for n in names:
            u = by_name.get(n)
            rows.append({
                "round": r, "site": n,
                "status": "active" if n in active else "dropped" if n in dropped else "skipped",
                "weight": weights.get(n, 0.0),
                "trust": info.get("trust", {}).get(n, np.nan),
                "quarantined": n in info.get("quarantined", []),
                "e_global": u.e_global if u else heartbeats.get(n, np.nan),
                "quality": u.quality if u else np.nan,
                "drift": n in new_drift,
                "faulty_active": clients[n].faulty_active,
                "val_mse_clean": val[n],
            })
        healthy = [n for n in names if n != faulty_name]
        rounds.append({"round": r, "n_active": len(active), "n_skipped": len(skipped),
                       "n_dropped": len(dropped), "aggregated": aggregated, "bytes": comm,
                       "val_mse_mean": float(np.mean(list(val.values()))),
                       "val_mse_healthy": float(np.mean([val[n] for n in healthy]))})
        if verbose:
            wtxt = " ".join(f"{n}:{weights.get(n, 0):.2f}" for n in names[:8])
            flags = (f" drift={sorted(new_drift)}" if new_drift else "") + \
                    (f" quarantined={info['quarantined']}" if info.get("quarantined") else "") + \
                    (f" dropped={dropped}" if dropped else "")
            print(f"    round {r:>2} | active {len(active):>3} | val_mse(healthy) "
                  f"{rounds[-1]['val_mse_healthy']:.5f} | w {wtxt}{flags}")

    name = method + ("_event" if event_aware else "")
    preds = {n: predict(global_model, clients[n].data.test) for n in names}   # clean test data
    return Result(name, preds, history=rows,
                  extra={"rounds": pd.DataFrame(rounds), "model_bytes": model_bytes,
                         "faulty_site": faulty_name})
