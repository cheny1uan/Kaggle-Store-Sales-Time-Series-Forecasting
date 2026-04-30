from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

from src.hybrid_v2.config import HybridV2Config
from src.hybrid_v2.features import add_calendar_features, merge_external_features
from src.hybrid_v2.io import load_tables, save_submission
from src.hybrid_v2.residual import fit_residual_model, get_feature_columns, predict_residuals
from src.hybrid_v2.rules import apply_zero_rule, build_zero_lookup
from src.hybrid_v2.trend import fit_trend_model, predict_trend


def add_lag_features(df: pd.DataFrame, history: pd.DataFrame | None, lags: tuple[int, ...], windows: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out["_target_row"] = 1
    out["_row_order"] = np.arange(len(out))
    if history is not None:
        base = history[["date", "store_nbr", "family", "sales", "transactions"]].copy()
        base["_target_row"] = 0
        base["_row_order"] = -1
        base = pd.concat([base, out], ignore_index=True, sort=False)
    else:
        base = out

    base["date"] = pd.to_datetime(base["date"])
    base = base.sort_values(["store_nbr", "family", "date", "_row_order"]).reset_index(drop=True)

    for lag in lags:
        base[f"sales_lag_{lag}"] = base.groupby(["store_nbr", "family"])["sales"].shift(lag)
    for window in windows:
        base[f"sales_roll_mean_{window}"] = base.groupby(["store_nbr", "family"])["sales"].transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        base[f"sales_roll_std_{window}"] = base.groupby(["store_nbr", "family"])["sales"].transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).std()
        )
    if history is not None:
        base = base[base["_target_row"] == 1].sort_values("_row_order").reset_index(drop=True)
    else:
        base = base.sort_values("_row_order").reset_index(drop=True)
    return base.drop(columns=["_target_row", "_row_order"]).fillna(-1)


def build_frame(tables: dict[str, pd.DataFrame], base: pd.DataFrame, history: pd.DataFrame | None, config: HybridV2Config) -> pd.DataFrame:
    frame = merge_external_features(base, tables["stores"], tables["oil"], tables["holidays"], tables["transactions"])
    frame = add_calendar_features(frame)
    frame = add_lag_features(frame, history=history, lags=config.lags, windows=config.rolling_windows)
    return frame


def temporal_split(train_df: pd.DataFrame, valid_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = pd.to_datetime(train_df["date"]).max()
    split_date = max_date - pd.Timedelta(days=valid_days)
    train_part = train_df[train_df["date"] <= split_date].copy()
    valid_part = train_df[train_df["date"] > split_date].copy()
    return train_part, valid_part


def recursive_predict(model_bundle, frame: pd.DataFrame, feature_cols: list[str], zero_lookup, config: HybridV2Config) -> pd.DataFrame:
    pred = frame.copy()
    pred["trend_pred"] = model_bundle["trend_pred"]
    pred["residual_pred"] = 0.0
    pred["sales_pred"] = pred["trend_pred"]
    pred = pred.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)
    pred["pred"] = np.nan
    pred["pred"] = model_bundle["final_pred"]
    pred["pred"] = np.clip(pred["pred"], 0, None)
    pred["pred"] = apply_zero_rule(pred[["store_nbr", "family", "onpromotion", "pred"]], zero_lookup)
    return pred


def run(mode: str = "fast") -> dict[str, object]:
    config = HybridV2Config(mode=mode)
    tables = load_tables(config.data_dir)
    train = tables["train"].copy()
    test = tables["test"].copy()
    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    train_part, valid_part = temporal_split(train, config.valid_days)
    if mode == "fast" and config.fast_history_days is not None:
        start = train_part["date"].max() - pd.Timedelta(days=config.fast_history_days)
        train_part = train_part[train_part["date"] > start].copy()

    train_frame = build_frame(tables, train_part, None, config)
    valid_frame = build_frame(tables, valid_part, train_frame, config)
    test_frame = build_frame(tables, test, train_frame, config)

    trend_model, trend_cols = fit_trend_model(train_frame, train_part["sales"], config.trend_fourier_orders, config.trend_alpha)
    train_frame["trend_pred"] = predict_trend(trend_model, train_frame, trend_cols, config.trend_fourier_orders)
    valid_frame["trend_pred"] = predict_trend(trend_model, valid_frame, trend_cols, config.trend_fourier_orders)
    test_frame["trend_pred"] = predict_trend(trend_model, test_frame, trend_cols, config.trend_fourier_orders)

    feature_cols = get_feature_columns(train_frame)
    residual_model, cat_cols, category_levels = fit_residual_model(
        train_frame,
        valid_frame,
        feature_cols,
        n_estimators=config.fast_n_estimators if mode == "fast" else config.full_n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        early_stopping_rounds=config.early_stopping_rounds,
    )

    zero_lookup = build_zero_lookup(train)
    valid_pred = np.expm1(np.log1p(valid_frame["trend_pred"].values) + predict_residuals(residual_model, valid_frame, feature_cols, category_levels))
    valid_pred = np.clip(valid_pred, 0, None)
    valid_pred = apply_zero_rule(valid_frame[["store_nbr", "family", "onpromotion"]].assign(pred=valid_pred), zero_lookup)
    score = mean_squared_log_error(valid_part["sales"], valid_pred) ** 0.5

    test_resid = predict_residuals(residual_model, test_frame, feature_cols, category_levels)
    test_pred = np.expm1(np.log1p(test_frame["trend_pred"].values) + test_resid)
    test_pred = np.clip(test_pred, 0, None)
    test_pred = apply_zero_rule(test_frame[["store_nbr", "family", "onpromotion"]].assign(pred=test_pred), zero_lookup)

    submission = tables["sample_submission"].copy()
    submission["sales"] = test_pred
    return {
        "config": asdict(config),
        "validation_rmsle": score,
        "submission": submission,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "trend_cols": trend_cols,
    }
