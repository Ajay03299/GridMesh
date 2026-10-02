"""Common-cloud and scarce-asset tests on one fixed learned forecast."""
import copy
from src.evaluation.serialization import dumps
from pathlib import Path

import numpy as np
import pandas as pd

from run_simulation import run_method
from src.data.adapter import load_config
from src.data.preprocessing import prepare_clients
from src.evaluation.experiments import predictions_frame
from src.reliability.safety import safe_forecast
from src.reserve.simulator import load_node, run_policies


def main():
    cfg = load_config()
    clients, _ = prepare_clients(cfg, 4, cfg["seed"])
    result = run_method("reliability_fedavg_event", clients, cfg, cfg["seed"], verbose=False)
    node = load_node(predictions_frame(result, clients))
    rows = []
    for scenario in ("healthy", "common_cloud_shock", "grid_import_reduced", "backup_depleted"):
        rc, stressed = copy.deepcopy(cfg), node.copy()
        rc["reserve"]["delta_sweep"] = [0.05]
        if scenario == "common_cloud_shock":
            midday = stressed.time.dt.hour.between(11, 13)
            stressed.loc[midday, "actual_mw"] *= 0.4
        elif scenario == "grid_import_reduced":
            rc["reserve"]["grid_import_limit_fraction"] = 0.4
        elif scenario == "backup_depleted":
            rc["reserve"]["backup_energy_hours"] = 0.5
        for _, summary in run_policies(stressed, rc, cfg["seed"]):
            rows.append({"scenario": scenario, **summary})
        print(f"finished {scenario}", flush=True)
    # Runtime input failures are exercised directly, without inventing a field outage.
    base = clients[0].test.base
    forecast_rows = []
    for scenario, kwargs in [("stale_inputs", {"stale_inputs": True}),
                              ("expired_forecast", {"age_minutes": 60}),
                              ("drift_alarm", {"drift_alarm": True}),
                              ("invalid_prediction", {})]:
        values = np.full_like(base, np.nan) if scenario == "invalid_prediction" else base
        prediction, health = safe_forecast(values, None, None, base, **kwargs)
        assert np.isfinite(prediction).all()
        forecast_rows.append({"scenario": scenario, **health})
    out = Path(cfg["paths"]["outputs"])
    pd.DataFrame(rows).to_csv(out / "tables/operational_stress.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(out / "tables/forecast_fallback_stress.csv", index=False)
    (out / "metrics/operational_stress.json").write_text(dumps(
        {"config": cfg, "seed": cfg["seed"], "results": rows, "fallbacks": forecast_rows}, indent=2))
    print(pd.DataFrame(rows)[["scenario", "policy", "shortfall_energy_mwh", "planned_capacity_gap_mwh"]])


if __name__ == "__main__":
    main()
