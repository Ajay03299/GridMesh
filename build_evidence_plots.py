"""Static scientific figures from completed enhanced experiment tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path("outputs")


def save(fig, name, caption):
    fig.text(.02, .015, caption, fontsize=8, color="#58716A")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(OUT / "plots" / name, dpi=180)
    plt.close(fig)


def main():
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(OUT / "tables/common_detail.csv")
    assert len(detail) == 60 and detail.seed.nunique() == 5
    frame = detail.groupby("method").global_rmse.agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(frame.index, frame["mean"], xerr=frame["std"], color="#4E7D61", capsize=3)
    ax.invert_yaxis()
    ax.set(xlabel="Daytime test RMSE (per-unit capacity)", title="12 methods: five-seed mean and standard deviation")
    save(fig, "enhanced_model_comparison.png", "Shared weather, modelled PV. Pooled methods centralize rows. Validation selects models, not these test ranks.")

    stress = pd.read_csv(OUT / "tables/reliability_stress_detail.csv")
    assert stress.seed.nunique() == 5 and len(stress) == 120
    grouped = stress.groupby(["scenario", "method"]).healthy_rmse.agg(["mean", "std"])
    scenarios = ["healthy", "stale_50pct", "dropout_50pct", "sensor_recovery", "seasonal_shift"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (method, color) in enumerate((("fedavg", "#58716A"),
            ("reliability_fedavg", "#4E7D61"), ("reliability_fedavg_event", "#D49A54"))):
        values = grouped.loc[[(s, method) for s in scenarios]]
        ax.bar(np.arange(len(scenarios)) + (i-1)*.25, values["mean"], width=.25,
               yerr=values["std"], capsize=3, color=color, label=method)
    ax.set(xticks=np.arange(len(scenarios)), xticklabels=scenarios,
           ylabel="Healthy-site test RMSE (per-unit capacity)", title="Fault and dropout comparisons")
    ax.legend(fontsize=9)
    save(fig, "enhanced_reliability.png", "Five paired seeds. Sensor faults are simulated. Trust protection and event skipping are distinct effects.")

    reserve = pd.read_csv(OUT / "tables/common_reserve_detail.csv")
    reserve = reserve[reserve.forecast_method == "reliability_fedavg"]
    assert reserve.seed.nunique() == 5 and len(reserve) == 20
    policies = ["fixed_20pct", "nsigma_d0.05", "guarded_nsigma_d0.05", "empirical_d0.05"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("reserve_energy_mwh", "shortfall_energy_mwh"),
                                 ("Scheduled backup", "Unserved energy")):
        values = reserve.groupby("policy")[metric].agg(["mean", "std"]).loc[policies]
        ax.bar(np.arange(4), values["mean"], yerr=values["std"], capsize=3, color="#4E7D61")
        ax.set(xticks=np.arange(4), xticklabels=["Fixed 20%", "Nominal", "Guarded", "Empirical"],
               ylabel="MWh", title=title)
    save(fig, "enhanced_reserve_comparison.png", "Five seeds, same MLP. Synthetic demand, assumed assets/costs, retrospective daily LP, daytime only.")
    print("Saved three enhanced evidence figures")


if __name__ == "__main__":
    main()
