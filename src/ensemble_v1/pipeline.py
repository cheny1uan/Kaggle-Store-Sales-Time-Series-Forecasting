from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import load_raw_data, merge_features
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


def recursive_forecast(
    base_df: pd.DataFrame,
    history_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    agg_source: pd.DataFrame,
    train_ref: pd.DataFrame,
    feature_cols: list[str],
    lgb_model,
    xgb_model,
    weights: dict[str, float],
    zero_set,
) -> pd.DataFrame:
    base = build_dataset(base_df, frames, agg_source)
    history = history_df[["date", "store_nbr", "family", "sales"]].copy()
    history["date"] = pd.to_datetime(history["date"])
    frames_out: list[pd.DataFrame] = []
    max_lookback = 70

    for current_date in sorted(pd.to_datetime(base["date"]).unique()):
        day_base = base[base["date"] == current_date].copy()
        recent_history = history[history["date"] >= pd.Timestamp(current_date) - pd.Timedelta(days=max_lookback)].copy()
        day_feat = add_time_series_features(day_base, history_df=recent_history)
        day_feat = day_feat.reindex(columns=["date", "id"] + feature_cols, fill_value=0)

        day_lgb = predict_lgb(lgb_model, train_ref, day_feat, feature_cols)
        day_xgb = predict_xgb(xgb_model, train_ref, day_feat, feature_cols)
        day_pred = blend_predictions({"lgb": day_lgb, "xgb": day_xgb}, weights)
        day_pred = apply_zero_forecast(day_feat[ZERO_KEY_COLS + ["onpromotion"]].assign(pred=day_pred), zero_set)
        day_pred = pd.Series(day_pred).clip(lower=0).to_numpy()

        day_out = day_feat[["id", "date", "store_nbr", "family", "onpromotion"]].copy()
        day_out["pred"] = day_pred
        frames_out.append(day_out)

        history_update = day_out[["date", "store_nbr", "family"]].copy()
        history_update["sales"] = day_pred
        history = pd.concat([history, history_update], ignore_index=True, sort=False)

    return pd.concat(frames_out, ignore_index=True, sort=False)


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
    agg_source = add_date_features(
        merge_features(
            base=train_part if mode == "fast" else train_raw,
            stores=frames["stores"],
            oil=frames["oil"],
            holidays=frames["holidays"],
            transactions=frames["transactions"],
        )
    )
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

    zero_set = build_zero_set(train_part if mode == "fast" else train_raw)
    print(f"Zero-sales store-family pairs: {len(zero_set):,}")

    valid_lgb_df = recursive_forecast(
        base_df=valid_part,
        history_df=train_part,
        frames=frames,
        agg_source=agg_source,
        train_ref=train_feat,
        feature_cols=feature_cols,
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        weights={"lgb": 1.0, "xgb": 0.0},
        zero_set=zero_set,
    )
    valid_xgb_df = recursive_forecast(
        base_df=valid_part,
        history_df=train_part,
        frames=frames,
        agg_source=agg_source,
        train_ref=train_feat,
        feature_cols=feature_cols,
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        weights={"lgb": 0.0, "xgb": 1.0},
        zero_set=zero_set,
    )
    rec_lgb_eval = valid_part[["id", "sales"]].merge(valid_lgb_df[["id", "pred"]], on="id", how="left")
    rec_xgb_eval = valid_part[["id", "sales"]].merge(valid_xgb_df[["id", "pred"]], on="id", how="left")
    rec_lgb_score = rmsle(rec_lgb_eval["sales"].values, rec_lgb_eval["pred"].fillna(0).values)
    rec_xgb_score = rmsle(rec_xgb_eval["sales"].values, rec_xgb_eval["pred"].fillna(0).values)
    weights = inverse_score_weights({"lgb": rec_lgb_score, "xgb": rec_xgb_score})
    if rec_xgb_score > rec_lgb_score * 1.15:
        print("Recursive XGBoost is much weaker than LightGBM; using LightGBM-only blend for submission.")
        weights = {"lgb": 1.0, "xgb": 0.0}
    print(f"Recursive LightGBM RMSLE: {rec_lgb_score:.5f}")
    print(f"Recursive XGBoost RMSLE: {rec_xgb_score:.5f}")
    print(f"Blend weights: {weights}")

    valid_pred_df = recursive_forecast(
        base_df=valid_part,
        history_df=train_part,
        frames=frames,
        agg_source=agg_source,
        train_ref=train_feat,
        feature_cols=feature_cols,
        lgb_model=lgb_model,
        xgb_model=xgb_model,
        weights=weights,
        zero_set=zero_set,
    )
    valid_eval = valid_part[["id", "sales"]].merge(valid_pred_df[["id", "pred"]], on="id", how="left")
    blend_score = rmsle(valid_eval["sales"].values, valid_eval["pred"].fillna(0).values)
    print(f"Recursive ensemble validation RMSLE: {blend_score:.5f}")

    if mode_cfg.full_retrain:
        full_agg_source = add_date_features(
            merge_features(
                base=train_raw,
                stores=frames["stores"],
                oil=frames["oil"],
                holidays=frames["holidays"],
                transactions=frames["transactions"],
            )
        )
        full_train_feat = build_dataset(train_raw, frames, full_agg_source)
        full_train_feat = add_time_series_features(full_train_feat)
        full_train_feat = full_train_feat.reindex(columns=["date", "sales"] + feature_cols, fill_value=0)

        final_lgb = fit_final_lgb_model(
            full_train_feat,
            feature_cols,
            n_estimators=getattr(lgb_model, "best_iteration_", None) or mode_cfg.lgb_estimators,
        )[0]
        final_xgb = fit_final_xgb_model(full_train_feat, feature_cols, n_estimators=mode_cfg.xgb_estimators)[0]
        test_pred_df = recursive_forecast(
            base_df=test_raw,
            history_df=train_raw,
            frames=frames,
            agg_source=full_agg_source,
            train_ref=full_train_feat,
            feature_cols=feature_cols,
            lgb_model=final_lgb,
            xgb_model=final_xgb,
            weights=weights,
            zero_set=zero_set,
        )
    else:
        test_pred_df = recursive_forecast(
            base_df=test_raw,
            history_df=train_part,
            frames=frames,
            agg_source=agg_source,
            train_ref=train_feat,
            feature_cols=feature_cols,
            lgb_model=lgb_model,
            xgb_model=xgb_model,
            weights=weights,
            zero_set=zero_set,
        )

    preds = test_pred_df.sort_values("id")["pred"].to_numpy()

    submission = frames["sample_submission"].copy()
    submission["sales"] = preds
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
