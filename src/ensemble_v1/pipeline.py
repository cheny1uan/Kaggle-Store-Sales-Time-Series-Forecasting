from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_raw_data
from src.ensemble_v1.config import EnsembleV1Config, MODES
from src.ensemble_v1.ensemble import blend_predictions, inverse_score_weights
from src.ensemble_v1.features import (
    ZERO_KEY_COLS,
    add_time_series_features,
    align_feature_frames,
    apply_zero_forecast,
    build_dataset,
    build_zero_set,
    prepare_time_split,
)
from src.ensemble_v1.models import (
    fit_final_lgb_model,
    fit_final_xgb_model,
    fit_lgb_model,
    fit_xgb_model,
    predict_lgb,
    predict_xgb,
    xgboost_available,
)
from src.features.make_features import add_date_features
from src.utils.metrics import rmsle


def _build_feature_sets(frames: dict[str, pd.DataFrame], train_part: pd.DataFrame, valid_part: pd.DataFrame, test_raw: pd.DataFrame, agg_source: pd.DataFrame, feature_drop: set[str]):
    train_feat = build_dataset(train_part, frames, agg_source)
    valid_feat = build_dataset(valid_part, frames, agg_source)
    test_feat = build_dataset(test_raw, frames, agg_source)
    train_feat = add_time_series_features(train_feat)
    valid_feat = add_time_series_features(valid_feat, history_df=train_feat)
    test_history = train_feat[["date", "store_nbr", "family", "sales", "transactions"]].copy()
    test_feat = add_time_series_features(test_feat, history_df=test_history)
    return align_feature_frames(train_feat, valid_feat, test_feat, feature_drop)


def run(mode: str = "fast") -> dict[str, object]:
    config = EnsembleV1Config(mode=mode)
    mode_cfg = MODES[mode]
    data_dir = Path(config.data_dir)
    output_dir = Path(config.output_dir)
    submission_dir = Path(config.submission_dir)
    output_dir.mkdir(exist_ok=True)
    submission_dir.mkdir(exist_ok=True)

    frames = load_raw_data(data_dir)
    train_raw = frames["train"].copy()
    test_raw = frames["test"].copy()
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    test_raw["date"] = pd.to_datetime(test_raw["date"])

    train_part, valid_part, split_date = prepare_time_split(train_raw, mode_cfg.valid_days, mode_cfg.history_days)
    agg_source = add_date_features(train_part if mode == "fast" else train_raw)
    train_feat, valid_feat, test_feat, feature_cols, removed_cols = _build_feature_sets(
        frames, train_part, valid_part, test_raw, agg_source, config.feature_drop
    )

    print(f"Mode: {mode}")
    print(f"Train rows: {len(train_part):,}")
    print(f"Valid rows: {len(valid_part):,}")
    print(f"Split date: {split_date.date()}")
    print(f"Dropped low-importance features: {removed_cols}")
    print(f"XGBoost installed: {xgboost_available()}")

    lgb_model, lgb_score, lgb_cat_cols = fit_lgb_model(
        train_feat,
        valid_feat,
        feature_cols,
        n_estimators=mode_cfg.lgb_estimators,
        early_stopping_rounds=mode_cfg.early_stopping,
    )
    print(f"LightGBM validation RMSLE: {lgb_score:.5f}")

    xgb_model, xgb_score, xgb_cat_cols = fit_xgb_model(
        train_feat,
        valid_feat,
        feature_cols,
        n_estimators=mode_cfg.xgb_estimators,
        early_stopping_rounds=mode_cfg.early_stopping,
    )
    print(f"XGBoost validation RMSLE: {xgb_score:.5f}")

    valid_lgb = predict_lgb(lgb_model, train_feat, valid_feat, feature_cols)
    valid_xgb = predict_xgb(xgb_model, train_feat, valid_feat, feature_cols)
    weights = inverse_score_weights({"lgb": lgb_score, "xgb": xgb_score})
    if xgb_score > lgb_score * 1.15:
        print("Second model is much weaker than LightGBM; using LightGBM-only blend for submission.")
        weights = {"lgb": 1.0, "xgb": 0.0}
    valid_blend = blend_predictions({"lgb": valid_lgb, "xgb": valid_xgb}, weights)
    blend_score = rmsle(valid_feat["sales"].values, valid_blend)
    print(f"Blend weights: {weights}")
    print(f"Ensemble validation RMSLE: {blend_score:.5f}")

    zero_set = build_zero_set(train_part if mode == "fast" else train_raw)
    print(f"Zero-sales store-family pairs: {len(zero_set):,}")

    if mode_cfg.full_retrain:
        full_agg_source = add_date_features(train_raw)
        full_train_feat = build_dataset(train_raw, frames, full_agg_source)
        full_train_feat = add_time_series_features(full_train_feat)
        full_train_feat = full_train_feat.reindex(columns=["date", "sales"] + feature_cols, fill_value=0)

        final_test_feat = build_dataset(test_raw, frames, full_agg_source)
        final_test_feat = add_time_series_features(
            final_test_feat,
            history_df=full_train_feat[["date", "store_nbr", "family", "sales", "transactions"]],
        )
        final_test_feat = final_test_feat.reindex(columns=["date", "id"] + feature_cols, fill_value=0)

        final_lgb = fit_final_lgb_model(full_train_feat, feature_cols, n_estimators=getattr(lgb_model, "best_iteration_", None) or mode_cfg.lgb_estimators)[0]
        final_xgb = fit_final_xgb_model(full_train_feat, feature_cols, n_estimators=mode_cfg.xgb_estimators)[0]
        test_lgb = predict_lgb(final_lgb, full_train_feat, final_test_feat, feature_cols)
        test_xgb = predict_xgb(final_xgb, full_train_feat, final_test_feat, feature_cols)
        pred_source = final_test_feat
    else:
        test_lgb = predict_lgb(lgb_model, train_feat, test_feat, feature_cols)
        test_xgb = predict_xgb(xgb_model, train_feat, test_feat, feature_cols)
        pred_source = test_feat

    preds = blend_predictions({"lgb": test_lgb, "xgb": test_xgb}, weights)
    pred_frame = pred_source[ZERO_KEY_COLS + ["onpromotion"]].copy()
    pred_frame["pred"] = preds
    preds = apply_zero_forecast(pred_frame, zero_set)
    preds = pd.Series(preds).clip(lower=0)

    submission = frames["sample_submission"].copy()
    submission["sales"] = preds.values
    out_path = submission_dir / mode_cfg.output_name
    submission.to_csv(out_path, index=False)

    return {
        "submission_path": out_path,
        "validation_rmsle": blend_score,
        "lgb_rmsle": lgb_score,
        "xgb_rmsle": xgb_score,
        "weights": weights,
        "feature_cols": feature_cols,
        "lgb_cat_cols": lgb_cat_cols,
        "xgb_cat_cols": xgb_cat_cols,
    }
