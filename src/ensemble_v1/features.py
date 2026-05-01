from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.load_data import merge_features
from src.features.make_features import (
    add_date_features,
    add_lag_rolling_features,
    apply_target_aggregates,
    build_target_aggregates,
)
from src.models.train_lgbm import get_feature_columns


ZERO_KEY_COLS = ["store_nbr", "family"]


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


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["oil_promo_interact"] = out["dcoilwtico"].fillna(0).astype("float32") * out["onpromotion"].fillna(0).astype("float32")
    if "family_sales_mean" in out.columns:
        out["promo_family_mean_interact"] = out["onpromotion"].fillna(0).astype("float32") * out["family_sales_mean"].fillna(0).astype("float32")
    if "store_nbr__family_sales_mean" in out.columns:
        out["promo_store_family_mean_interact"] = out["onpromotion"].fillna(0).astype("float32") * out["store_nbr__family_sales_mean"].fillna(0).astype("float32")
    return out


def build_dataset(base_df: pd.DataFrame, frames: dict[str, pd.DataFrame], agg_source_df: pd.DataFrame) -> pd.DataFrame:
    base = merge_features(
        base=base_df,
        stores=frames["stores"],
        oil=frames["oil"],
        holidays=frames["holidays"],
        transactions=frames["transactions"],
    )
    base = add_date_features(base)
    agg_maps = build_target_aggregates(agg_source_df)
    base = apply_target_aggregates(base, agg_maps)
    base = add_interaction_features(base)
    return base


def add_time_series_features(target_df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
    sales_history = None
    transactions_history = None
    if history_df is not None:
        sales_cols = [c for c in ["date", "store_nbr", "family", "sales"] if c in history_df.columns]
        transactions_cols = [c for c in ["date", "store_nbr", "transactions"] if c in history_df.columns]
        sales_history = history_df[sales_cols].copy() if sales_cols else None
        transactions_history = history_df[transactions_cols].copy() if transactions_cols else None

    out = add_lag_rolling_features(
        target_df,
        history_df=sales_history,
        group_cols=["store_nbr", "family"],
        value_col="sales",
        feature_prefix="sales",
    )
    out = add_lag_rolling_features(
        out,
        history_df=transactions_history,
        group_cols=["store_nbr"],
        value_col="transactions",
        feature_prefix="transactions",
    )
    return out


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

