"""PHASE 1 — inspect the dataset before building anything.

Usage:
    python inspect_data.py
    python inspect_data.py --path data/raw/other_file.csv
"""
import argparse

import pandas as pd

from src.data.adapter import load_config, load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--path", help="override data.path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.path:
        cfg["data"]["path"] = args.path

    df, meta = load_dataset(cfg)
    ts = df["timestamp"]
    step = ts.diff().dropna()

    print("=" * 60)
    print(f"FILE          : {cfg['data']['path']}")
    print(f"SHAPE         : {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"TIMESTAMP     : {meta.timestamp_col}")
    print(f"COVERAGE      : {ts.min()}  ->  {ts.max()}  ({(ts.max() - ts.min()).days} days)")
    print(f"RESOLUTION    : most common step = {step.mode().iloc[0]}")
    print(f"GAPS          : {(step != step.mode().iloc[0]).sum():,} irregular steps")
    print(f"DUPLICATE TS  : {ts.duplicated().sum():,}")
    print(f"RAW LABEL     : {meta.target_col}  (task = {meta.task})")
    print(f"MEASURED POWER: {meta.has_real_power}")
    mode = cfg.get("target", {}).get("mode", "column")
    if mode == "pv_from_irradiance":
        print("FORECAST TARGET (config): PV power MODELED from the real GHI + temperature "
              "(simplified PVWatts) -> used for FL and the reserve simulation")
    else:
        print(f"FORECAST TARGET (config): column '{cfg['target'].get('column')}'")

    print("\nCOLUMNS (dtype, % missing):")
    for c in df.columns:
        print(f"  {c:<28} {str(df[c].dtype):<16} {df[c].isna().mean() * 100:6.2f}%")

    print("\nRAW LABEL DISTRIBUTION:")
    if not meta.target_col:
        print("  (data.target_col not set)")
    elif meta.task == "classification":
        counts = df[meta.target_col].value_counts().sort_index()
        pct = (counts / counts.sum() * 100).round(2)
        print(pd.DataFrame({"count": counts, "pct": pct}).to_string())
    else:
        print(df[meta.target_col].describe().to_string())

    print("\nCANDIDATE NUMERIC FEATURES:", meta.feature_cols)
    print("=" * 60)


if __name__ == "__main__":
    main()
