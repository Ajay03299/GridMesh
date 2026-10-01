"""Dataset adapter: load any renewable/weather CSV into a clean, time-indexed DataFrame.

Handles both:
  * wind/weather data with a class target (e.g. WindSpeed_Class) -> classification
  * solar/wind data with continuous generation (e.g. PV power)    -> regression

Nothing here invents data. It only loads, detects columns and reports.
"""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import pandas as pd
import yaml

TIME_NAME_HINTS = ("timestamp", "datetime", "date_time", "time", "date")
# Some datasets split time into separate columns instead of one timestamp.
SPLIT_TIME_PARTS = ["year", "month", "day", "hour", "minute"]


@dataclass
class DatasetMeta:
    timestamp_col: str
    target_col: str
    task: str                      # "classification" | "regression"
    power_col: Optional[str] = None
    site_col: Optional[str] = None
    feature_cols: list = field(default_factory=list)
    has_real_power: bool = False   # False => reserve sim must use a SYNTHETIC power series


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _read_any(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Put your file there or change data.path in config.yaml."
        )
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _detect_timestamp(df, requested, assumed_year=2019):
    """Return (df_with_'timestamp'_column, source_description).

    If the file has Month/Day/Hour/Minute but no Year, `assumed_year` is used
    (an explicit, logged assumption — it only affects the calendar, not the data).
    """
    if requested and requested != "auto":
        df["timestamp"] = pd.to_datetime(df[requested])
        return df, requested

    # 1) a single column whose name looks like time
    for col in df.columns:
        if any(h in col.lower() for h in TIME_NAME_HINTS):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.95:
                df["timestamp"] = parsed
                return df, col

    # 2) separate Year/Month/Day/Hour/Minute columns
    lower = {c.lower(): c for c in df.columns}
    if all(p in lower for p in ("month", "day")):
        parts = {p: df[lower[p]] for p in SPLIT_TIME_PARTS if p in lower}
        source = "+".join(lower[p] for p in parts)
        if "year" not in parts:
            parts["year"] = assumed_year
            source += f" (year ASSUMED = {assumed_year})"
        df["timestamp"] = pd.to_datetime(pd.DataFrame(parts))
        return df, source

    raise ValueError("Could not detect a timestamp. Set data.timestamp_col in config.yaml.")


def _detect_task(series, requested):
    if requested in ("classification", "regression"):
        return requested
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    # Integer-valued with few distinct values -> treat as classes
    s = series.dropna()
    if (s == s.round()).all() and s.nunique() <= 20:
        return "classification"
    return "regression"


def load_dataset(cfg):
    """Load the dataset described in cfg['data']. Returns (df, DatasetMeta)."""
    dcfg = cfg["data"]
    df = _read_any(dcfg["path"])
    df.columns = [str(c).strip() for c in df.columns]

    df, ts_source = _detect_timestamp(df, dcfg.get("timestamp_col", "auto"),
                                      dcfg.get("assumed_year", 2019))
    df = df.sort_values("timestamp").reset_index(drop=True)

    target = dcfg.get("target_col")          # optional raw label, only shown by inspect_data.py
    if target and target not in df.columns:
        raise ValueError(f"data.target_col '{target}' not in columns: {list(df.columns)}")

    power_col = dcfg.get("power_col") or (cfg.get("target") or {}).get("column")
    site_col = dcfg.get("site_col")
    time_part_cols = {p.split(" ")[0] for p in ts_source.split("+")}
    excluded = {"timestamp", target, power_col, site_col} | time_part_cols
    features = [c for c in df.columns
                if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]

    meta = DatasetMeta(
        timestamp_col=ts_source,
        target_col=target,
        task=_detect_task(df[target], dcfg.get("task", "auto")) if target else "regression",
        power_col=power_col,
        site_col=site_col,
        feature_cols=features,
        has_real_power=power_col is not None and power_col in df.columns,
    )
    return df, meta
