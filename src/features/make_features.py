from __future__ import annotations

import numpy as np
import pandas as pd


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out["day"] = out["date"].dt.day.astype("int8")
    out["dayofweek"] = out["date"].dt.dayofweek.astype("int8")
    out["dayofyear"] = out["date"].dt.dayofyear.astype("int16")
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype("int16")
    out["quarter"] = out["date"].dt.quarter.astype("int8")
    out["is_month_start"] = out["date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")
    days_in_month = out["date"].dt.days_in_month.astype("int16")
    day = out["day"].astype("int16")
    out["is_payday"] = ((day == 15) | (day == days_in_month)).astype("int8")
    out["days_to_next_payday"] = np.where(
        day == 15,
        0,
        np.where(
            day == days_in_month,
            0,
            np.where(day < 15, np.minimum(15 - day, days_in_month - day), days_in_month - day),
        ),
    ).astype("int8")
    out["days_since_prev_payday"] = np.where(
        day == days_in_month,
        0,
        np.where(day >= 15, day - 15, day),
    ).astype("int8")
    out["payday_window_3"] = (
        (out["days_to_next_payday"] <= 3) | (out["days_since_prev_payday"] <= 3)
    ).astype("int8")
    earthquake_date = pd.Timestamp("2016-04-16")
    delta_days = (out["date"] - earthquake_date).dt.days
    out["is_earthquake_day"] = (delta_days == 0).astype("int8")
    out["earthquake_window_7"] = (delta_days.between(0, 7)).astype("int8")
    out["earthquake_window_30"] = (delta_days.between(0, 30)).astype("int8")
    out["earthquake_post_impact"] = (delta_days.between(0, 21)).astype("int8")
    angle = 2 * np.pi * out["dayofyear"] / 365.25
    out["doy_sin"] = np.sin(angle).astype("float32")
    out["doy_cos"] = np.cos(angle).astype("float32")
    return out


def build_target_aggregates(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    agg_specs = {
        "store_nbr": ["store_nbr"],
        "family": ["family"],
        "store_nbr__family": ["store_nbr", "family"],
        "family__dayofweek": ["family", "dayofweek"],
        "store_nbr__dayofweek": ["store_nbr", "dayofweek"],
        "type": ["type"],
        "city": ["city"],
        "state": ["state"],
        "family__month": ["family", "month"],
        "store_nbr__month": ["store_nbr", "month"],
    }
    mappings: dict[str, pd.DataFrame] = {}
    for name, cols in agg_specs.items():
        grp = train_df.groupby(cols)["sales"].agg(["mean", "std"]).reset_index()
        grp.columns = cols + [f"{name}_sales_mean", f"{name}_sales_std"]
        mappings[name] = grp
    return mappings


def apply_target_aggregates(df: pd.DataFrame, mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    for name, mapping in mappings.items():
        key_cols = name.split("__")
        out = out.merge(mapping, on=key_cols, how="left")
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    group_cols: list[str] | None = None,
    value_col: str = "sales",
    feature_prefix: str | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["store_nbr", "family"]
    feature_prefix = feature_prefix or value_col
    lags = lags or [1, 7, 14, 28]
    windows = windows or [7, 28]

    target = df.copy()
    target["_is_target"] = 1
    target["_order"] = np.arange(len(target))

    if history_df is None:
        work = target.copy()
    else:
        needed_cols = list(dict.fromkeys(group_cols + ["date", value_col]))
        available_cols = [c for c in needed_cols if c in history_df.columns]
        history = history_df[available_cols].copy()
        history["_is_target"] = 0
        history["_order"] = -1
        work = pd.concat([history, target], ignore_index=True, sort=False)

    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(group_cols + ["date", "_order"]).reset_index(drop=True)

    grouped = work.groupby(group_cols, sort=False)[value_col]
    for lag in lags:
        work[f"{feature_prefix}_lag_{lag}"] = grouped.transform(lambda s, lag=lag: s.shift(lag))

    for window in windows:
        work[f"{feature_prefix}_roll_mean_{window}"] = grouped.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        work[f"{feature_prefix}_roll_std_{window}"] = grouped.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).std()
        )

    feat_cols = [
        c
        for c in work.columns
        if c.startswith(f"{feature_prefix}_lag_") or c.startswith(f"{feature_prefix}_roll_")
    ]
    work[feat_cols] = work[feat_cols].fillna(-1)

    if history_df is None:
        return work.sort_values("_order").drop(columns=["_is_target", "_order"])

    return work[work["_is_target"] == 1].sort_values("_order").drop(columns=["_is_target", "_order"])
