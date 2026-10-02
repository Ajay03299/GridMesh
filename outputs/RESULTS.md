# GridMesh: verified model, reliability and scale evidence

Five paired simulation seeds (42–46), one shared 52,560-row weather record, 30-minute horizon.
Daytime test RMSE is in per-unit installed capacity. Each site uses a train-fitted standardizer.
These are simulated outcomes, not independent locations or measured customer outages.

## Model comparison

| method | rmse_mean | rmse_std | mae_mean | bias_mean | worst_site_rmse | validation_rmse | runtime_seconds | model_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| centralized | 0.0642 | 0.0008 | 0.0363 | -0.0018 | 0.0666 | 0.0618 | 32.2498 | 31748.0000 |
| centralized_xgboost | 0.0652 | 0.0011 | 0.0359 | -0.0015 | 0.0674 | 0.0612 | 14.5996 | 652808.2000 |
| decision_tree | 0.0699 | 0.0026 | 0.0379 | -0.0016 | 0.0729 | 0.0661 | 4.4042 | 22455.4000 |
| fedavg | 0.0641 | 0.0012 | 0.0365 | -0.0028 | 0.0667 | 0.0615 | 46.1384 | 31748.0000 |
| gru | 0.0654 | 0.0023 | 0.0376 | -0.0022 | 0.0682 | 0.0622 | 303.5305 | 77572.0000 |
| local_only | 0.0667 | 0.0025 | 0.0388 | -0.0019 | 0.0713 | 0.0637 | 33.0329 | 126992.0000 |
| local_xgboost | 0.0665 | 0.0014 | 0.0362 | -0.0011 | 0.0693 | 0.0618 | 22.1572 | 2087510.2000 |
| lstm | 0.0651 | 0.0017 | 0.0372 | -0.0024 | 0.0678 | 0.0621 | 210.7243 | 96004.0000 |
| persistence | 0.0908 | 0.0015 | 0.0673 | -0.0020 | 0.0941 | 0.0875 | 0.0004 | nan |
| reliability_fedavg | 0.0640 | 0.0010 | 0.0360 | -0.0018 | 0.0664 | 0.0614 | 46.4821 | 31748.0000 |
| reliability_fedavg_event | 0.0648 | 0.0012 | 0.0369 | -0.0034 | 0.0674 | 0.0620 | 24.1577 | 31748.0000 |
| smart_persistence | 0.0735 | 0.0011 | 0.0362 | 0.0000 | 0.0761 | 0.0683 | 0.0013 | nan |

Pooled XGBoost and pooled MLP centralize raw rows. Local XGBoost retains local rows but does
not learn across owners. GRU/LSTM rows use the same reliability-aware FL protocol as MLP.
Validation mean ranks centralized_xgboost first in this run. Tree early stopping uses validation
residual RMSE. Neural stopping uses daytime validation forecast MSE. Test metrics are common,
but these distinct stopping objectives and training budgets are documented limitations.
Size uses neural parameter bytes versus serialized tree bytes. Runtime is measured on this
Windows machine during concurrent experiments and is indicative, not an isolated latency SLA.

## Reliability scorecard

| scenario | method | healthy_rmse | worst_site_rmse | forecast_bias | fallback_rounds | quarantined_sites |
| --- | --- | --- | --- | --- | --- | --- |
| biased_25pct | fedavg | 0.0635 | 0.0665 | -0.0024 | 0.0000 | 0.0000 |
| biased_25pct | reliability_fedavg | 0.0637 | 0.0666 | -0.0024 | 0.0000 | 1.0000 |
| biased_25pct | reliability_fedavg_event | 0.0640 | 0.0670 | -0.0043 | 9.0000 | 1.0000 |
| dropout_50pct | fedavg | 0.0646 | 0.0669 | -0.0023 | 7.2000 | 0.0000 |
| dropout_50pct | reliability_fedavg | 0.0643 | 0.0664 | -0.0026 | 7.2000 | 0.0000 |
| dropout_50pct | reliability_fedavg_event | 0.0650 | 0.0679 | -0.0048 | 13.0000 | 0.0000 |
| fault_and_dropout | fedavg | 0.0643 | 0.0673 | -0.0037 | 13.6000 | 0.0000 |
| fault_and_dropout | reliability_fedavg | 0.0643 | 0.0673 | -0.0041 | 1.4000 | 1.0000 |
| fault_and_dropout | reliability_fedavg_event | 0.0649 | 0.0679 | -0.0053 | 12.6000 | 1.0000 |
| healthy | fedavg | 0.0641 | 0.0667 | -0.0028 | 0.0000 | 0.0000 |
| healthy | reliability_fedavg | 0.0640 | 0.0664 | -0.0018 | 0.0000 | 0.0000 |
| healthy | reliability_fedavg_event | 0.0648 | 0.0674 | -0.0034 | 6.6000 | 0.0000 |
| seasonal_shift | fedavg | 0.0541 | 0.0670 | 0.0118 | 0.0000 | 0.0000 |
| seasonal_shift | reliability_fedavg | 0.0546 | 0.0670 | 0.0095 | 0.0000 | 0.0000 |
| seasonal_shift | reliability_fedavg_event | 0.0512 | 0.0592 | 0.0070 | 5.8000 | 0.0000 |
| sensor_recovery | fedavg | 0.0638 | 0.0667 | -0.0031 | 0.0000 | 0.0000 |
| sensor_recovery | reliability_fedavg | 0.0641 | 0.0669 | -0.0028 | 0.0000 | 1.0000 |
| sensor_recovery | reliability_fedavg_event | 0.0640 | 0.0668 | -0.0036 | 8.6000 | 1.0000 |
| stale_50pct | fedavg | 0.0689 | 0.0730 | 0.0078 | 0.0000 | 0.0000 |
| stale_50pct | reliability_fedavg | 0.0628 | 0.0672 | -0.0031 | 0.0000 | 0.0000 |
| stale_50pct | reliability_fedavg_event | 0.0632 | 0.0677 | -0.0042 | 7.2000 | 0.0000 |
| under_participation | fedavg | 0.0648 | 0.0677 | -0.0019 | 7.8000 | 0.0000 |
| under_participation | reliability_fedavg | 0.0650 | 0.0677 | -0.0005 | 7.8000 | 0.0000 |
| under_participation | reliability_fedavg_event | 0.0665 | 0.0694 | -0.0022 | 16.8000 | 0.0000 |

For two severe stale-sensor sites, reliability-aware FL improves healthy-site RMSE by
8.9% relative to FedAvg in this paired experiment. Results depend on the fault type
and baseline. No universal fault-tolerance claim follows. Recovery probes admit healed clients
after healthy reports. Fault recovery and dropout runs also retain all healthy evaluation sites.
Clean test features/targets isolate training/update damage. Direct runtime input-failure tests
are separate, rather than a claim of an end-to-end live sensor-fault replay.
Recovery means the first positive aggregation weight after the sensor heals at round 10. It
does not mean forecast-error recovery or wall-clock availability. An absent recovery is NaN,
rather than a fabricated success. The separate five-seed replay uses the same frozen settings.

| seed | method | quarantine_weight_recovery_rounds |
| --- | --- | --- |
| 42 | fedavg | 0.0000 |
| 42 | reliability_fedavg | 2.0000 |
| 42 | reliability_fedavg_event | 2.0000 |
| 43 | fedavg | 0.0000 |
| 43 | reliability_fedavg | 2.0000 |
| 43 | reliability_fedavg_event | 5.0000 |
| 44 | fedavg | 0.0000 |
| 44 | reliability_fedavg | 2.0000 |
| 44 | reliability_fedavg_event | nan |
| 45 | fedavg | 0.0000 |
| 45 | reliability_fedavg | 2.0000 |
| 45 | reliability_fedavg_event | nan |
| 46 | fedavg | 0.0000 |
| 46 | reliability_fedavg | 2.0000 |
| 46 | reliability_fedavg_event | 5.0000 |

## Scale and traffic

| n_sites | method | global_rmse | sites_per_round | comm_mb_per_round | feeder_comm_mb_per_round | seconds | process_peak_memory_mb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | fedavg | 0.0655 | 4.0000 | 0.3812 | 0.3812 | 17.7000 | 525.1441 |
| 4 | reliability_fedavg_event | 0.0669 | 3.0000 | 0.2859 | 0.2859 | 7.0000 | 525.1441 |
| 20 | fedavg | 0.0665 | 20.0000 | 1.9062 | 1.9062 | 40.3000 | 982.7983 |
| 20 | reliability_fedavg_event | 0.0680 | 11.0000 | 1.0485 | 1.0485 | 27.1000 | 982.7983 |
| 50 | fedavg | 0.0707 | 20.0000 | 1.9062 | 1.9062 | 4.9000 | 982.7983 |
| 50 | reliability_fedavg_event | 0.0710 | 11.3333 | 1.0808 | 1.0808 | 3.5000 | 982.7983 |
| 100 | fedavg | 0.0711 | 20.0000 | 1.9062 | 1.9062 | 7.5000 | 982.7983 |
| 100 | reliability_fedavg_event | 0.0719 | 11.3333 | 1.0816 | 1.0816 | 6.0000 | 982.7983 |
| 100 | hierarchical_reliability_fedavg | 0.0711 | 20.0000 | 2.2874 | 0.3812 | 5.7000 | 982.7983 |
| 250 | fedavg | 0.0708 | 20.0000 | 1.9062 | 1.9062 | 8.8000 | 982.7983 |
| 250 | reliability_fedavg_event | 0.0706 | 11.3333 | 1.0840 | 1.0840 | 10.3000 | 982.7983 |
| 250 | hierarchical_reliability_fedavg | 0.0708 | 20.0000 | 2.7639 | 0.8578 | 8.4000 | 982.7983 |
| 500 | fedavg | 0.0710 | 20.0000 | 1.9062 | 1.9062 | 14.1000 | 982.7983 |
| 500 | reliability_fedavg_event | 0.0705 | 11.3333 | 1.0880 | 1.0880 | 16.7000 | 982.7983 |
| 500 | hierarchical_reliability_fedavg | 0.0710 | 20.0000 | 3.2087 | 1.3025 | 13.5000 | 982.7983 |

At 100 logical clients, event participation reduces traffic 43.3% with a
+1.0% RMSE change. At the same scale, hierarchy reduces feeder-boundary traffic
80.0%. Total hierarchical traffic includes the extra neighbourhood/feeder hop.
This does not demonstrate better accuracy or lower total traffic than flat FL.

Sizes 4/20 use full virtual-site datasets. Sizes 50/100/250/500 reuse four compact reference
datasets (2,000 training and 1,000 validation/test rows each). This deliberately measures
logical-client scheduling, payload accounting and runtime. Geographic generalization remains
unmeasured. Three rounds and one seed are used at scale. Memory is the cumulative process peak,
so the 20-site full dataset may determine later peaks. Traffic is a modeled payload count,
including current model, local update, candidate-validation model and scalar metadata, but
excluding transport framing/TLS, authentication, retries and actual network latency.

## Forecast-to-reserve comparison

| forecast_method | policy | reserve_energy_mwh | shortfall_energy_mwh | availability_pct | planned_capacity_gap_mwh | total_cost |
| --- | --- | --- | --- | --- | --- | --- |
| local_xgboost | empirical_d0.05 | 1609.1783 | 282.1988 | 85.3210 | 1059.3579 | 7253.1552 |
| local_xgboost | fixed_20pct | 2420.6400 | 313.7025 | 81.6790 | 2497.0331 | 8694.6908 |
| local_xgboost | guarded_nsigma_d0.05 | 1838.2292 | 291.1412 | 84.9136 | 1048.6298 | 7661.0524 |
| local_xgboost | nsigma_d0.05 | 1574.8528 | 275.0354 | 86.4506 | 628.8595 | 7075.5618 |
| reliability_fedavg | empirical_d0.05 | 1646.7337 | 263.0323 | 85.2160 | 973.5143 | 6907.3794 |
| reliability_fedavg | fixed_20pct | 2420.6400 | 310.0090 | 79.8827 | 2502.2905 | 8620.8196 |
| reliability_fedavg | guarded_nsigma_d0.05 | 1896.3109 | 280.8448 | 83.7037 | 1121.8685 | 7513.2075 |
| reliability_fedavg | nsigma_d0.05 | 1575.5243 | 278.0981 | 85.1667 | 575.6029 | 7137.4862 |
| reliability_fedavg_event | empirical_d0.05 | 1632.1687 | 254.6347 | 85.7654 | 939.5957 | 6724.8621 |
| reliability_fedavg_event | fixed_20pct | 2420.6400 | 299.3146 | 80.1358 | 2503.6342 | 8406.9313 |
| reliability_fedavg_event | guarded_nsigma_d0.05 | 1865.7856 | 275.2851 | 84.1173 | 1054.6065 | 7371.4869 |
| reliability_fedavg_event | nsigma_d0.05 | 1557.7483 | 265.6442 | 85.8580 | 560.7056 | 6870.6332 |
| smart_persistence | empirical_d0.05 | 1663.9287 | 325.1844 | 80.5494 | 1281.6938 | 8167.6161 |
| smart_persistence | fixed_20pct | 2420.6400 | 314.0404 | 77.3519 | 2512.0743 | 8701.4472 |
| smart_persistence | guarded_nsigma_d0.05 | 1832.3919 | 314.6481 | 79.9383 | 1185.6848 | 8125.3535 |
| smart_persistence | nsigma_d0.05 | 1701.1756 | 291.3743 | 81.9506 | 858.4666 | 7528.6608 |

For reliability-aware MLP, the nominal n-sigma policy changes scheduled backup by
-34.9%, unserved energy by -10.3%
and assumed total cost by -17.2% versus its fixed-20% margin baseline.
The guarded policy is reported separately: it can consume scarce backup energy earlier and
may worsen aggregate shortfalls. More margin is not automatically better when energy binds.
The nominal 95% target applies to the forecast-error event, never annual feeder uptime.

The daily LP is a retrospective schedule benchmark using that day's sequence of rolling forecasts.
Its margins and monitors are causal, but the full-day allocation is not a deployable day-ahead
planner. A pilot needs either genuine forecasts issued before day start or rolling optimization
with residual energy state. Cash tariffs and emissions are not measured. Cost units use assumed
backup and shortfall coefficients 1 and 20 per MWh. No rupee or carbon-saving claim follows.

## Engineering targets: measured rather than guaranteed

Operational shortages use one fixed event-aware forecast and seed 42. The cloud shock reduces
midday actual solar without changing the issued forecast. Other tests reduce imports or backup
energy. These are distinct scenarios, not five-seed operational guarantees. The guarded margin
violation rate is a forecast-error metric, separate from service availability.

| scenario | policy | margin_violation_rate | shortfall_energy_mwh | planned_capacity_gap_mwh | availability_pct |
| --- | --- | --- | --- | --- | --- |
| healthy | fixed_20pct | nan | 328.9600 | 2538.4130 | 76.4815 |
| healthy | nsigma_d0.05 | nan | 277.4484 | 622.7643 | 85.4012 |
| healthy | guarded_nsigma_d0.05 | 0.0583 | 293.8639 | 1113.3719 | 83.7654 |
| healthy | empirical_d0.05 | nan | 279.3996 | 1062.5326 | 84.9691 |
| common_cloud_shock | fixed_20pct | nan | 2550.9049 | 2538.4130 | 59.7222 |
| common_cloud_shock | nsigma_d0.05 | nan | 2491.2223 | 5996.3691 | 61.0802 |
| common_cloud_shock | guarded_nsigma_d0.05 | 0.1568 | 2494.2823 | 6826.6935 | 60.9259 |
| common_cloud_shock | empirical_d0.05 | nan | 1781.6504 | 7734.1196 | 57.0988 |
| grid_import_reduced | fixed_20pct | nan | 2452.2793 | 7082.4125 | 33.9815 |
| grid_import_reduced | nsigma_d0.05 | nan | 2452.3334 | 4334.2644 | 34.5062 |
| grid_import_reduced | guarded_nsigma_d0.05 | 0.0583 | 2475.9858 | 5095.4458 | 34.6605 |
| grid_import_reduced | empirical_d0.05 | nan | 2442.9191 | 4831.8755 | 34.6296 |
| backup_depleted | fixed_20pct | nan | 471.6447 | 4686.2930 | 61.7593 |
| backup_depleted | nsigma_d0.05 | nan | 473.3732 | 1941.3861 | 65.4321 |
| backup_depleted | guarded_nsigma_d0.05 | 0.0583 | 478.6470 | 2699.3264 | 61.6667 |
| backup_depleted | empirical_d0.05 | nan | 462.2720 | 2443.0933 | 69.1667 |

| target | method | observed | unit | passed |
| --- | --- | --- | --- | --- |
| 50% dropout <=2% RMSE degradation | fedavg | 0.8522 | relative percent | True |
| 50% dropout <=2% RMSE degradation | reliability_fedavg | 0.5629 | relative percent | True |
| 50% dropout <=2% RMSE degradation | reliability_fedavg_event | 0.2307 | relative percent | True |
| 100-site traffic reduction >=40% | event-aware | 43.2589 | percent | True |
| 100-site accuracy degradation <=2% | event-aware | 1.0304 | relative percent | True |
| fault10: less healthy-site damage than FedAvg | reliability_fedavg | 1.3261 | damage difference, percentage points | False |
| fault20: less healthy-site damage than FedAvg | reliability_fedavg | -0.7436 | damage difference, percentage points | True |

Failed targets stay visible. In particular, additional event skipping can worsen learning with
few clients and heavy dropout. A future availability-aware sampling policy needs validation on
new data. Insufficient assets also remain a visible capacity gap instead of a false success.
Fault damage normalizes each method to its own healthy run. The reported difference is
trust-aware damage minus FedAvg damage, so negative values indicate less damage. These
100-client fault targets use one paired seed, unlike the five-seed four-site stress matrix.

## Reproduction and current scope

Run `python audit.py`, `python run_common_comparison.py`, `python run_tree_comparison.py`,
`python run_reliability_stress.py`, `python run_scaling.py --max-sites 500`,
`python run_scale_stress.py`, `python run_operational_stress.py`, then `python build_evidence_report.py`.
Quick modes save separate files. See docs/MODEL_RELIABILITY_SCALABILITY.md for methodology.

The automatic audit passes 36 checks. All clients/coordinators execute in one process. Raw
histories remain local by protocol convention, without secure aggregation or formal privacy.
The supplied weather file lacks verified year/location metadata. Year 2019 is assumed, and an
Indian location is not asserted. Field validation requires measured Indian multi-site PV/load,
asset limits, data permissions and a DISCOM/operator shadow pilot.
