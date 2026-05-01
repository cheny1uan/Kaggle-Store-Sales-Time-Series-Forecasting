from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

from src.hybrid_v2.config import HybridV2Config
from src.hybrid_v2.features import add_calendar_features, merge_external_features
from src.hybrid_v2.io import load_tables
from src.hybrid_v2.residual import fit_residual_model, get_feature_columns, predict_residuals
from src.hybrid_v2.rules import apply_zero_rule, build_zero_lookup
from src.hybrid_v2.trend import fit_trend_model, predict_trend


def add_lag_features(df: pd.DataFrame, history: pd.DataFrame | None, lags: tuple[int, ...], windows: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy()
    out["_target_row"] = 1
    out["_row_order"] = np.arange(len(out))
    if history is not None:
        base = history[["date", "store_nbr", "family", "sales"]].copy()
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
    base = base.drop(columns=["_target_row", "_row_order"])
    numeric_cols = base.select_dtypes(include=[np.number]).columns
    object_cols = base.select_dtypes(include=["object", "category", "string"]).columns
    base[numeric_cols] = base[numeric_cols].fillna(-1)
    base[object_cols] = base[object_cols].fillna("NA")
    return base


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


def recursive_forecast(
    tables: dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
    initial_history: pd.DataFrame,
    trend_model,
    trend_cols: list[str],
    residual_model,
    feature_cols: list[str],
    category_levels: dict[str, pd.Index],
    zero_lookup: pd.MultiIndex,
    config: HybridV2Config,
) -> pd.DataFrame:
    """Predict one calendar day at a time and feed predictions back as history."""
    history = initial_history[["date", "store_nbr", "family", "sales"]].copy()
    history["date"] = pd.to_datetime(history["date"])
    frames: list[pd.DataFrame] = []

    work = merge_external_features(base_df, tables["stores"], tables["oil"], tables["holidays"], tables["transactions"])
    work = add_calendar_features(work)
    work["date"] = pd.to_datetime(work["date"])
    max_lookback = max(max(config.lags), max(config.rolling_windows)) + 7
    for current_date in sorted(work["date"].unique()):
        day_base = work[work["date"] == current_date].copy()
        recent_history = history[history["date"] >= pd.Timestamp(current_date) - pd.Timedelta(days=max_lookback)].copy()
        day_frame = add_lag_features(day_base, recent_history, config.lags, config.rolling_windows)
        day_frame["trend_pred"] = predict_trend(trend_model, day_frame, trend_cols, config.trend_fourier_orders)

        resid = predict_residuals(residual_model, day_frame, feature_cols, category_levels)
        pred = np.expm1(np.log1p(np.clip(day_frame["trend_pred"].to_numpy(), 0, None)) + resid)
        pred = np.clip(pred, 0, None)
        pred = apply_zero_rule(day_frame[["store_nbr", "family", "onpromotion"]].assign(pred=pred), zero_lookup)

        day_frame["pred"] = pred
        frames.append(day_frame)

        history_update = day_frame[["date", "store_nbr", "family"]].copy()
        history_update["sales"] = pred
        history = pd.concat([history, history_update], ignore_index=True, sort=False)

    return pd.concat(frames, ignore_index=True, sort=False)


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
    valid_frame = build_frame(tables, valid_part, train_part, config)

    trend_model, trend_cols = fit_trend_model(train_frame, train_part["sales"], config.trend_fourier_orders, config.trend_alpha)
    train_frame["trend_pred"] = predict_trend(trend_model, train_frame, trend_cols, config.trend_fourier_orders)
    valid_frame["trend_pred"] = predict_trend(trend_model, valid_frame, trend_cols, config.trend_fourier_orders)

    feature_cols = [c for c in get_feature_columns(train_frame) if c not in config.feature_drop]
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
    valid_forecast = recursive_forecast(
        tables=tables,
        base_df=valid_part,
        initial_history=train_part,
        trend_model=trend_model,
        trend_cols=trend_cols,
        residual_model=residual_model,
        feature_cols=feature_cols,
        category_levels=category_levels,
        zero_lookup=zero_lookup,
        config=config,
    )
    valid_eval = valid_part[["id", "sales"]].merge(valid_forecast[["id", "pred"]], on="id", how="left")
    score = mean_squared_log_error(valid_eval["sales"], valid_eval["pred"].fillna(0)) ** 0.5

    test_forecast = recursive_forecast(
        tables=tables,
        base_df=test,
        initial_history=train,
        trend_model=trend_model,
        trend_cols=trend_cols,
        residual_model=residual_model,
        feature_cols=feature_cols,
        category_levels=category_levels,
        zero_lookup=zero_lookup,
        config=config,
    )

    submission = tables["sample_submission"].copy()
    submission = submission[["id"]].merge(test_forecast[["id", "pred"]], on="id", how="left")
    submission["sales"] = submission["pred"].fillna(0)
    submission = submission.drop(columns=["pred"])
    return {
        "config": asdict(config),
        "validation_rmsle": score,
        "submission": submission,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "trend_cols": trend_cols,
    }
