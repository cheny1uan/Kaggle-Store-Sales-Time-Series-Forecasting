from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import build_holiday_features, build_oil_features, merge_features
from src.features.make_features import (
    add_date_features,
    add_lag_rolling_features,
    apply_target_aggregates,
    build_target_aggregates,
)
from src.models.train_lgbm import get_feature_columns


ZERO_KEY_COLS = ["store_nbr", "family"]
LONG_SALES_LAGS = [1, 7, 14, 28, 56, 91, 182, 364, 365, 728]
REFERENCE_SALES_LAGS = [1, 7, 14, 28, 56, 91, 182, 364, 365, 728, 1095]
BASE_SALES_LAGS = [1, 7, 14, 28]
SALES_WINDOWS = [7, 28]
LONG_SALES_WINDOWS = [7, 28, 56]
TRANSACTION_LAGS = [1, 7, 14, 28]
REFERENCE_TRANSACTION_LAGS = [1, 7, 14, 16, 17, 18, 19, 20, 21, 22, 28]
TRANSACTION_WINDOWS = [7, 28]
PROMOTION_LAGS = [1, 7]
PROMOTION_WINDOWS = [7, 28]


def prepare_time_split(train_raw: pd.DataFrame, valid_days: int, history_days: int | None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    train_raw = train_raw.copy()
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    max_date = train_raw["date"].max()
    split_date = max_date - pd.Timedelta(days=valid_days)
    valid_part = train_raw[train_raw["date"] > split_date].copy()
    train_part = train_raw[train_raw["date"] <= split_date].copy()

    if history_days is not None:
        history_start = split_date - pd.Timedelta(days=history_days)
        train_part = train_part[train_part["date"] > history_start].copy()

    return train_part, valid_part, split_date


def add_localized_holiday_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    hol = holidays.copy()
    hol["date"] = pd.to_datetime(hol["date"])
    hol["transferred"] = hol["transferred"].astype(bool)
    active = hol[~hol["transferred"]].copy()

    national = active[active["locale"] == "National"].groupby("date", as_index=False).size()
    national = national.rename(columns={"size": "active_holiday_national"})
    regional = active[active["locale"] == "Regional"].groupby(["date", "locale_name"], as_index=False).size()
    regional = regional.rename(columns={"locale_name": "state", "size": "active_holiday_regional"})
    local = active[active["locale"] == "Local"].groupby(["date", "locale_name"], as_index=False).size()
    local = local.rename(columns={"locale_name": "city", "size": "active_holiday_local"})
    events = active[active["type"] == "Event"].groupby("date", as_index=False).size()
    events = events.rename(columns={"size": "active_event_count"})

    out = out.merge(national, on="date", how="left")
    out = out.merge(regional, on=["date", "state"], how="left")
    out = out.merge(local, on=["date", "city"], how="left")
    out = out.merge(events, on="date", how="left")

    cols = [
        "active_holiday_national",
        "active_holiday_regional",
        "active_holiday_local",
        "active_event_count",
    ]
    out[cols] = out[cols].fillna(0).astype("int8")
    out["active_holiday_any_scoped"] = (
        out["active_holiday_national"]
        + out["active_holiday_regional"]
        + out["active_holiday_local"]
        > 0
    ).astype("int8")
    return out


def _normalize_holiday_description(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.lower()
        .str.replace(r"[+-]\d+", "", regex=True)
        .str.replace(r"\b(de|del|traslado|recupero|puente|-)\b", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def add_named_holiday_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    hol = holidays.copy()
    hol["date"] = pd.to_datetime(hol["date"])
    hol["transferred"] = hol["transferred"].astype(bool)
    active = hol[(hol["locale"] == "National") & (~hol["transferred"])].copy()
    active["description_norm"] = _normalize_holiday_description(active["description"])
    active["named_holiday_earthquake"] = active["description_norm"].str.contains("terremoto", regex=False).astype("int8")
    active["named_holiday_christmas"] = active["description_norm"].str.contains("navidad", regex=False).astype("int8")
    active["named_holiday_mothers_day"] = active["description_norm"].str.contains("dia la madre", regex=False).astype("int8")
    active["named_holiday_labor_day"] = active["description_norm"].str.contains("dia trabajo", regex=False).astype("int8")
    active["named_holiday_new_year"] = active["description_norm"].str.contains("primer dia ano", regex=False).astype("int8")
    active["named_holiday_soccer"] = active["description_norm"].str.contains("futbol", regex=False).astype("int8")
    active["named_holiday_dead_day"] = active["description_norm"].str.contains("dia difuntos", regex=False).astype("int8")
    active["named_holiday_black_friday"] = active["description_norm"].str.contains("black friday", regex=False).astype("int8")
    active["named_holiday_cyber_monday"] = active["description_norm"].str.contains("cyber monday", regex=False).astype("int8")

    cols = [c for c in active.columns if c.startswith("named_holiday_")]
    daily = active.groupby("date", as_index=False)[cols].max()
    out = out.merge(daily, on="date", how="left")
    out[cols] = out[cols].fillna(0).astype("int8")
    out["named_holiday_peak"] = (
        out[
            [
                "named_holiday_earthquake",
                "named_holiday_christmas",
                "named_holiday_mothers_day",
                "named_holiday_labor_day",
                "named_holiday_new_year",
                "named_holiday_soccer",
                "named_holiday_dead_day",
            ]
        ].sum(axis=1)
        > 0
    ).astype("int8")
    return out


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["oil_promo_interact"] = out["dcoilwtico"].fillna(0).astype("float32") * out["onpromotion"].fillna(0).astype("float32")
    out["onpromotion_log1p"] = np.log1p(out["onpromotion"].fillna(0).clip(lower=0)).astype("float32")
    out["is_promo"] = (out["onpromotion"].fillna(0).astype(float) > 0).astype("int8")
    out["promo_date_sum"] = out.groupby("date")["onpromotion"].transform("sum").fillna(0).astype("float32")
    out["promo_store_date_sum"] = out.groupby(["date", "store_nbr"])["onpromotion"].transform("sum").fillna(0).astype("float32")
    out["promo_family_date_sum"] = out.groupby(["date", "family"])["onpromotion"].transform("sum").fillna(0).astype("float32")
    out["promo_store_date_share"] = (
        out["onpromotion"].fillna(0).astype("float32")
        / (out["promo_store_date_sum"].replace(0, np.nan).astype("float32"))
    ).fillna(0).astype("float32")
    out["promo_family_date_share"] = (
        out["onpromotion"].fillna(0).astype("float32")
        / (out["promo_family_date_sum"].replace(0, np.nan).astype("float32"))
    ).fillna(0).astype("float32")
    if "family_sales_mean" in out.columns:
        out["promo_family_mean_interact"] = out["onpromotion"].fillna(0).astype("float32") * out["family_sales_mean"].fillna(0).astype("float32")
    if "store_nbr__family_sales_mean" in out.columns:
        out["promo_store_family_mean_interact"] = out["onpromotion"].fillna(0).astype("float32") * out["store_nbr__family_sales_mean"].fillna(0).astype("float32")
    return out


def build_sales_dataset(
    base_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    agg_source_df: pd.DataFrame,
    transactions_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = merge_features(
        base=base_df,
        stores=frames["stores"],
        oil=frames["oil"],
        holidays=frames["holidays"],
        transactions=frames["transactions"],
        transactions_override=transactions_override,
    )
    base = add_localized_holiday_features(base, frames["holidays"])
    base = add_named_holiday_features(base, frames["holidays"])
    base = add_date_features(base)
    agg_maps = build_target_aggregates(agg_source_df)
    base = apply_target_aggregates(base, agg_maps)
    base = add_interaction_features(base)
    return base


def build_transaction_base_features(base_df: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    base = base_df.copy()
    base["date"] = pd.to_datetime(base["date"])
    if "transactions" not in base.columns:
        base["transactions"] = 0.0
    base = base.merge(frames["stores"], on="store_nbr", how="left")
    base = base.merge(build_oil_features(frames["oil"]), on="date", how="left")
    base = base.merge(build_holiday_features(frames["holidays"]), on="date", how="left")
    base = add_localized_holiday_features(base, frames["holidays"])
    base = add_named_holiday_features(base, frames["holidays"])
    base = add_date_features(base)
    return base


def build_transaction_aggregates(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    agg_specs = {
        "store_nbr": ["store_nbr"],
        "store_nbr__dayofweek": ["store_nbr", "dayofweek"],
        "store_nbr__month": ["store_nbr", "month"],
        "city": ["city"],
        "state": ["state"],
        "type": ["type"],
        "cluster": ["cluster"],
    }
    mappings: dict[str, pd.DataFrame] = {}
    for name, cols in agg_specs.items():
        grp = train_df.groupby(cols)["transactions"].agg(["mean", "std"]).reset_index()
        grp.columns = cols + [f"{name}_tx_mean", f"{name}_tx_std"]
        mappings[name] = grp
    return mappings


def apply_transaction_aggregates(df: pd.DataFrame, mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    for name, mapping in mappings.items():
        key_cols = name.split("__")
        out = out.merge(mapping, on=key_cols, how="left")
    return out


def add_sales_time_series_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    use_long_lags: bool = False,
    use_reference_features: bool = False,
) -> pd.DataFrame:
    sales_history = None
    transactions_history = None
    promotion_history = None
    if history_df is not None:
        sales_cols = [c for c in ["date", "store_nbr", "family", "sales"] if c in history_df.columns]
        transactions_cols = [c for c in ["date", "store_nbr", "transactions"] if c in history_df.columns]
        promotion_cols = [c for c in ["date", "store_nbr", "family", "onpromotion"] if c in history_df.columns]
        sales_history = history_df[sales_cols].copy() if sales_cols else None
        transactions_history = history_df[transactions_cols].copy() if transactions_cols else None
        promotion_history = history_df[promotion_cols].copy() if promotion_cols else None

    sales_lags = BASE_SALES_LAGS
    sales_windows = SALES_WINDOWS
    transaction_lags = TRANSACTION_LAGS
    if use_reference_features:
        sales_lags = REFERENCE_SALES_LAGS
        sales_windows = LONG_SALES_WINDOWS
        transaction_lags = REFERENCE_TRANSACTION_LAGS
    elif use_long_lags:
        sales_lags = LONG_SALES_LAGS
        sales_windows = LONG_SALES_WINDOWS

    out = add_lag_rolling_features(
        target_df,
        history_df=sales_history,
        group_cols=["store_nbr", "family"],
        value_col="sales",
        feature_prefix="sales",
        lags=sales_lags,
        windows=sales_windows,
    )
    out = add_lag_rolling_features(
        out,
        history_df=transactions_history,
        group_cols=["store_nbr"],
        value_col="transactions",
        feature_prefix="transactions",
        lags=transaction_lags,
        windows=TRANSACTION_WINDOWS,
    )
    if use_reference_features:
        out = add_lag_rolling_features(
            out,
            history_df=promotion_history,
            group_cols=["store_nbr", "family"],
            value_col="onpromotion",
            feature_prefix="promotion",
            lags=PROMOTION_LAGS,
            windows=PROMOTION_WINDOWS,
        )
    return out


def add_transaction_time_series_features(target_df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    tx_history = None
    if history_df is not None:
        cols = [c for c in ["date", "store_nbr", "transactions"] if c in history_df.columns]
        tx_history = history_df[cols].copy() if cols else None
    return add_lag_rolling_features(
        target_df,
        history_df=tx_history,
        group_cols=["store_nbr"],
        value_col="transactions",
        feature_prefix="transactions",
    )


def align_feature_frames(
    train_feat: pd.DataFrame,
    valid_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    feature_drop: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    all_feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in all_feature_cols if c not in feature_drop]
    removed_cols = [c for c in all_feature_cols if c in feature_drop]

    train_feat = train_feat.reindex(columns=["date", "sales"] + feature_cols, fill_value=0)
    valid_feat = valid_feat.reindex(columns=["date", "sales"] + feature_cols, fill_value=0)
    test_feat = test_feat.reindex(columns=["date", "id"] + feature_cols, fill_value=0)
    return train_feat, valid_feat, test_feat, feature_cols, removed_cols


def build_zero_set(train_df: pd.DataFrame) -> pd.MultiIndex:
    zero_pairs = train_df.groupby(ZERO_KEY_COLS)["sales"].sum().reset_index()
    zero_pairs = zero_pairs[zero_pairs["sales"] == 0][ZERO_KEY_COLS]
    return pd.MultiIndex.from_frame(zero_pairs)


def build_recent_zero_set(train_df: pd.DataFrame, recent_days: int = 21) -> pd.MultiIndex:
    if train_df.empty:
        return pd.MultiIndex.from_frame(pd.DataFrame(columns=ZERO_KEY_COLS))

    work = train_df.copy()
    work["date"] = pd.to_datetime(work["date"])
    anchor_date = work["date"].max()
    start_date = anchor_date - pd.Timedelta(days=recent_days - 1)
    recent = work[work["date"] >= start_date].copy()
    zero_pairs = recent.groupby(ZERO_KEY_COLS)["sales"].sum().reset_index()
    zero_pairs = zero_pairs[zero_pairs["sales"] == 0][ZERO_KEY_COLS]
    return pd.MultiIndex.from_frame(zero_pairs)


def apply_zero_forecast(pred_df: pd.DataFrame, zero_set: pd.MultiIndex) -> np.ndarray:
    out = pred_df.copy()
    pair_index = pd.MultiIndex.from_frame(out[ZERO_KEY_COLS])
    zero_mask = pair_index.isin(zero_set)
    if "onpromotion" in out.columns:
        zero_mask = zero_mask & (out["onpromotion"].fillna(0).astype(float) <= 0)
    preds = out["pred"].to_numpy(copy=True)
    preds[zero_mask] = 0.0
    print(f"Zero forecasting overrides: {int(zero_mask.sum()):,} rows")
    return preds
