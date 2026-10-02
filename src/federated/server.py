"""Synchronous federated training loop (simulation of one server + N sites).

Per round:
  1. select clients (sampling, event-aware skipping, random dropout)
  2. active clients receive the global model, report e_global + quality, train locally
  3. skipping clients send a tiny heartbeat (error of their last received model) if event-aware
  4. server aggregates (FedAvg or reliability-aware) -> new global model
  5. log weights, trust, drift, participation, bytes, per-site validation error

Model selection (same idea as early stopping in the baselines): the clients' reported
validation errors of the model they RECEIVED are averaged with the same weights the method
uses for aggregation; the server keeps the global model with the lowest such score.
No test data and no oracle knowledge of which site is faulty is used.
"""
import numpy as np
import pandas as pd
import time

from src.evaluation.experiments import Result
from src.federated.client import FLClient
from src.federated.client_selection import select_clients
from src.federated.fedavg import FedAvg, weighted_average
from src.federated.reliability_aware import ReliabilityAwareFedAvg
from src.federated.hierarchical import HierarchicalReliabilityFedAvg
from src.models.forecasting_model import build_model, get_params, mse_on, n_bytes, predict, set_params
from src.reliability.drift import DriftDetector
from src.reliability.safety import screen_updates, should_rollback, validation_gate
from src.reliability.safety import safe_forecast
from src.evaluation.metrics import smart_persistence_forecast

METADATA_BYTES = 64   # sample counts, quality/errors and candidate-validation gate scalars
MODEL_TRANSFERS = 3  # current-model download, local-update upload, candidate-validation download


def run_federated(clients_data, cfg, seed, method="fedavg", event_aware=False,
                  faulty_clients=(), verbose=True):
    fcfg, ecfg, scfg = cfg["federated"], cfg["event_aware"], cfg.get("safety", {})
    rng = np.random.default_rng([seed, 7])
    clients = {c.params.name: FLClient(c, cfg, seed) for c in clients_data}
    names = list(clients)
    faulty_names = {clients_data[i].params.name for i in (faulty_clients or ())}

    global_model = build_model(clients_data[0].train.X.shape[1], cfg["model"], seed)
    gp = get_params(global_model)
    model_bytes = n_bytes(gp)
    last_seen = {n: gp for n in names}
    last_seen_aggs = {n: 0 for n in names}   # how many aggregations the model a site last saw had
    agg = FedAvg() if method == "fedavg" else ReliabilityAwareFedAvg(cfg["reliability"])
    if method == "hierarchical_reliability_fedavg":
        agg = HierarchicalReliabilityFedAvg(cfg["reliability"], names,
                                            fcfg.get("neighbourhood_size", 25))
    detector = DriftDetector(ecfg["drift_k"], ecfg["window"], ecfg["min_history"],
                             ecfg.get("min_rel_increase", 0.0))
    drifting, rows, rounds = set(), [], []
    best_score, best_params, best_round = np.inf, gp, 0
    trusted_score, trusted_params = np.inf, gp
    client_seconds = coordinator_seconds = 0.0
    quarantined = set()

    for r in range(1, fcfg["rounds"] + 1):
        if r >= cfg["faults"]["start_round"]:
            for fn in faulty_names:
                clients[fn].faulty_active = True
        if r >= cfg["faults"].get("recovery_round", np.inf):
            for fn in faulty_names:
                clients[fn].faulty_active = False

        active, skipped, dropped = select_clients(names, rng, fcfg, ecfg if event_aware else None,
                                                  drifting, r, quarantined)
        started = time.perf_counter()
        raw_updates = [clients[n].fit(gp, r) for n in active]
        client_seconds += time.perf_counter() - started
        started = time.perf_counter()
        updates, rejected, update_norms = screen_updates(
            raw_updates, gp, scfg.get("max_update_norm_ratio", np.inf))
        for n in active:
            last_seen[n], last_seen_aggs[n] = gp, r - 1

        heartbeats = {}
        if event_aware:   # everyone who did not train and did not drop out reports its error
            for n in names:
                if n not in active and n not in dropped:
                    if n in fcfg.get("unavailable_sites", []):
                        continue
                    heartbeats[n] = clients[n].evaluate(last_seen[n])[0]
        errors = {u.name: u.e_global for u in updates} | heartbeats
        new_drift = {n for n, e in errors.items() if detector.update(n, e)} if event_aware else set()
        drifting = new_drift

        aggregated = len(updates) >= fcfg["min_clients"]
        weights, info = ({}, {})
        fallback = None if aggregated else ("last_trusted_model" if raw_updates else "no_updates")
        rollback_reason = ""
        received = gp                       # the model the clients just evaluated
        if aggregated:
            # a heartbeat only informs trust if it scores a model that was trained at least once
            # (otherwise a site that skipped early looks bad just for holding the untrained model)
            peer = {n: e for n, e in errors.items() if n in heartbeats and last_seen_aggs[n] >= 1}
            weights, info = agg.weights(updates, peer_errors=peer)
            # Preserve exclusion until a recovery probe produces healthy trust.
            for u in updates:
                if u.name in info.get("quarantined", []):
                    quarantined.add(u.name)
                else:
                    quarantined.discard(u.name)
            if sum(w > 0 for w in weights.values()) < fcfg["min_clients"]:
                weights = {u.name: 0.0 for u in updates}
                aggregated = False
                fallback = "last_trusted_model"
            candidate = (weighted_average([u.params for u in updates],
                                          [weights[u.name] for u in updates]) if aggregated else gp)
            set_params(global_model, candidate)
            # Clients score their own available data. Clean data below is reporting-only.
            reports = {u.name: clients[u.name].validation_report(candidate)
                       for u in updates if weights[u.name] > 0}
            candidate_val = (sum(weights[n] * report["mse"] for n, report in reports.items())
                             if aggregated else trusted_score)
            rollback, rollback_reason = should_rollback(
                candidate_val, trusted_score, scfg.get("rollback_val_ratio", np.inf))
            valid, gate_reason = validation_gate(list(reports.values()), scfg)
            if not valid:
                rollback, rollback_reason = True, gate_reason
            if rollback:
                gp, fallback = trusted_params, "last_trusted_model"
                aggregated = False
            else:
                if aggregated:
                    gp, trusted_params, trusted_score = candidate, candidate, candidate_val
            fallback = fallback or info.get("fallback")
        # validation score of the RECEIVED model, weighted like this method's aggregation
        score = np.nan
        if updates and not rollback_reason:
            sw = weights if weights else {u.name: u.n_samples for u in updates}
            tot = sum(sw[u.name] for u in updates)
            if tot > 0:
                score = sum(sw[u.name] * u.e_global for u in updates) / tot
                if fcfg.get("select_best", True) and score < best_score:
                    best_score, best_params, best_round = score, received, r - 1
        set_params(global_model, gp)

        # server-side bookkeeping for plots: global model on every site's CLEAN validation data
        val = {n: mse_on(global_model, clients[n].data.val) for n in names}
        site_comm = len(active) * (MODEL_TRANSFERS * model_bytes + METADATA_BYTES) + \
            len(heartbeats) * ecfg["heartbeat_bytes"]
        feeder_comm = (info.get("reporting_groups", 0) *
                       (MODEL_TRANSFERS * model_bytes + METADATA_BYTES)
                       if method == "hierarchical_reliability_fedavg" else site_comm)
        comm = site_comm + feeder_comm if method == "hierarchical_reliability_fedavg" else site_comm
        by_name = {u.name: u for u in raw_updates}
        for n in names:
            u = by_name.get(n)
            rows.append({
                "round": r, "site": n,
                "status": "active" if n in active else "dropped" if n in dropped else "skipped",
                "weight": weights.get(n, 0.0),
                "trust": info.get("trust", {}).get(n, np.nan),
                "quarantined": n in quarantined,
                "quarantine_reason": ("trust_below_threshold" if n in quarantined else ""),
                "rejected": n in rejected,
                "rejection_reason": rejected.get(n, ""),
                "update_norm": update_norms.get(n, np.nan),
                "e_global": u.e_global if u else heartbeats.get(n, np.nan),
                "quality": u.quality if u else np.nan,
                "drift": n in new_drift,
                "faulty_active": clients[n].faulty_active,
                "val_mse_clean": val[n],
            })
        healthy = [n for n in names if n not in faulty_names]
        rounds.append({"round": r, "n_active": len(active), "n_skipped": len(skipped),
                       "n_dropped": len(dropped), "n_accepted": len(updates),
                       "n_rejected": len(rejected), "aggregated": aggregated, "bytes": comm,
                       "fallback": fallback or "", "rollback_reason": rollback_reason,
                       "site_bytes": site_comm, "feeder_bytes": feeder_comm,
                       "reporting_neighbourhoods": info.get("reporting_groups", 0),
                       "val_reported": score,
                       "val_mse_mean": float(np.mean(list(val.values()))),
                       "val_mse_healthy": float(np.mean([val[n] for n in healthy]))})
        if verbose:
            wtxt = " ".join(f"{n}:{weights.get(n, 0):.2f}" for n in names[:8])
            flags = (f" drift={sorted(new_drift)}" if new_drift else "") + \
                    (f" quarantined={info['quarantined']}" if info.get("quarantined") else "") + \
                    (f" dropped={dropped}" if dropped else "")
            flags += (f" rejected={rejected}" if rejected else "")
            flags += (f" fallback={fallback}" if fallback else "")
            print(f"    round {r:>2} | active {len(active):>3} | val_mse(healthy) "
                  f"{rounds[-1]['val_mse_healthy']:.5f} | w {wtxt}{flags}")
        coordinator_seconds += time.perf_counter() - started

    if fcfg.get("select_best", True):
        set_params(global_model, best_params)
    if verbose:
        print(f"    selected global model after round {best_round} (reported val {best_score:.5f})")
    name = method + ("_event" if event_aware else "")
    started = time.perf_counter()
    preds = {n: predict(global_model, clients[n].data.test) for n in names}   # clean test data
    vpreds = {n: predict(global_model, clients[n].data.val) for n in names}
    forecast_health = {}
    for n in names:
        age = 60 if n in fcfg.get("unavailable_sites", []) else 0
        preds[n], forecast_health[n] = safe_forecast(
            preds[n] if best_round > 0 else None, None, None,
            smart_persistence_forecast(clients[n].data.test), age_minutes=age)
        vpreds[n], _ = safe_forecast(vpreds[n] if best_round > 0 else None, None, None,
                                   smart_persistence_forecast(clients[n].data.val), age_minutes=age)
    inference_seconds = time.perf_counter() - started
    return Result(name, preds, history=rows, val_preds=vpreds,
                  extra={"rounds": pd.DataFrame(rounds), "model_bytes": model_bytes,
                         "faulty_sites": sorted(faulty_names), "best_round": best_round,
                         "training_seconds": client_seconds, "coordinator_seconds": coordinator_seconds,
                         "inference_seconds": inference_seconds, "forecast_health": forecast_health})
