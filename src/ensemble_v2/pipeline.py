from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.load_data import load_raw_data, merge_features
from src.ensemble_v2.config import EnsembleV2Config, MODES
from src.ensemble_v2.ensemble import blend_predictions, inverse_score_weights
from src.ensemble_v2.features import (
    ZERO_KEY_COLS,
    add_time_series_features,
    align_feature_frames,
    apply_zero_forecast,
    build_dataset,
    build_zero_set,
    prepare_time_split,
)
from src.ensemble_v2.models import (
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
    models: dict[str, object],
    model_types: dict[str, str],
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

        day_predictions = {}
        for name, model in models.items():
            if weights.get(name, 0.0) == 0.0:
                continue
            if model_types[name] == "lgb":
                day_predictions[name] = predict_lgb(model, train_ref, day_feat, feature_cols)
            elif model_types[name] == "xgb":
                day_predictions[name] = predict_xgb(model, train_ref, day_feat, feature_cols)
            else:
                raise ValueError(f"Unknown model type: {model_types[name]}")
        day_pred = blend_predictions(day_predictions, weights)
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
    config = EnsembleV2Config(mode=mode)
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

    models: dict[str, object] = {}
    model_types: dict[str, str] = {}
    batch_scores: dict[str, float] = {}
    lgb_best_iterations: dict[str, int] = {}
    lgb_cat_cols: list[str] = []
    xgb_cat_cols: list[str] = []

    for seed in mode_cfg.lgb_seeds:
        name = f"lgb_{seed}"
        model, score, lgb_cat_cols = fit_lgb_model(
            train_feat,
            valid_feat,
            feature_cols,
            n_estimators=mode_cfg.lgb_estimators,
            early_stopping_rounds=mode_cfg.early_stopping,
            seed=seed,
        )
        models[name] = model
        model_types[name] = "lgb"
        batch_scores[name] = score
        lgb_best_iterations[name] = getattr(model, "best_iteration_", None) or mode_cfg.lgb_estimators
        print(f"LightGBM seed {seed} validation RMSLE: {score:.5f}")

    xgb_model, xgb_score, xgb_cat_cols = fit_xgb_model(
        train_feat,
        valid_feat,
        feature_cols,
        n_estimators=mode_cfg.xgb_estimators,
        early_stopping_rounds=mode_cfg.early_stopping,
        seed=mode_cfg.xgb_seed,
    )
    models[f"xgb_{mode_cfg.xgb_seed}"] = xgb_model
    model_types[f"xgb_{mode_cfg.xgb_seed}"] = "xgb"
    batch_scores[f"xgb_{mode_cfg.xgb_seed}"] = xgb_score
    print(f"XGBoost seed {mode_cfg.xgb_seed} validation RMSLE: {xgb_score:.5f}")

    zero_set = build_zero_set(train_part if mode == "fast" else train_raw)
    print(f"Zero-sales store-family pairs: {len(zero_set):,}")

    recursive_scores: dict[str, float] = {}
    for name in models:
        one_hot_weights = {model_name: 0.0 for model_name in models}
        one_hot_weights[name] = 1.0
        pred_df = recursive_forecast(
            base_df=valid_part,
            history_df=train_part,
            frames=frames,
            agg_source=agg_source,
            train_ref=train_feat,
            feature_cols=feature_cols,
            models=models,
            model_types=model_types,
            weights=one_hot_weights,
            zero_set=zero_set,
        )
        eval_df = valid_part[["id", "sales"]].merge(pred_df[["id", "pred"]], on="id", how="left")
        recursive_scores[name] = rmsle(eval_df["sales"].values, eval_df["pred"].fillna(0).values)
        print(f"Recursive {name} RMSLE: {recursive_scores[name]:.5f}")

    best_recursive = min(recursive_scores.values())
    active_scores = {
        name: score
        for name, score in recursive_scores.items()
        if score <= best_recursive * 1.03
    }
    weights = inverse_score_weights(active_scores)
    weights = {name: weights.get(name, 0.0) for name in models}
    print(f"Blend weights: {weights}")

    valid_pred_df = recursive_forecast(
        base_df=valid_part,
        history_df=train_part,
        frames=frames,
        agg_source=agg_source,
        train_ref=train_feat,
        feature_cols=feature_cols,
        models=models,
        model_types=model_types,
        weights=weights,
        zero_set=zero_set,
    )
    valid_eval = valid_part[["id", "sales"]].merge(valid_pred_df[["id", "pred"]], on="id", how="left")
    blend_score = rmsle(valid_eval["sales"].values, valid_eval["pred"].fillna(0).values)
    print(f"Recursive ensemble validation RMSLE: {blend_score:.5f}")
    if blend_score > best_recursive:
        best_name = min(recursive_scores, key=recursive_scores.get)
        weights = {name: 0.0 for name in models}
        weights[best_name] = 1.0
        valid_pred_df = recursive_forecast(
            base_df=valid_part,
            history_df=train_part,
            frames=frames,
            agg_source=agg_source,
            train_ref=train_feat,
            feature_cols=feature_cols,
            models=models,
            model_types=model_types,
            weights=weights,
            zero_set=zero_set,
        )
        valid_eval = valid_part[["id", "sales"]].merge(valid_pred_df[["id", "pred"]], on="id", how="left")
        blend_score = rmsle(valid_eval["sales"].values, valid_eval["pred"].fillna(0).values)
        print(f"Blend underperformed; using best single model {best_name}.")
        print(f"Final recursive validation RMSLE: {blend_score:.5f}")

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

        final_models: dict[str, object] = {}
        final_model_types: dict[str, str] = {}
        for seed in mode_cfg.lgb_seeds:
            name = f"lgb_{seed}"
            if weights.get(name, 0.0) == 0.0:
                continue
            final_models[name] = fit_final_lgb_model(
                full_train_feat,
                feature_cols,
                n_estimators=lgb_best_iterations[name],
                seed=seed,
            )[0]
            final_model_types[name] = "lgb"
        xgb_name = f"xgb_{mode_cfg.xgb_seed}"
        if weights.get(xgb_name, 0.0) > 0.0:
            final_models[xgb_name] = fit_final_xgb_model(
                full_train_feat,
                feature_cols,
                n_estimators=mode_cfg.xgb_estimators,
                seed=mode_cfg.xgb_seed,
            )[0]
            final_model_types[xgb_name] = "xgb"
        test_pred_df = recursive_forecast(
            base_df=test_raw,
            history_df=train_raw,
            frames=frames,
            agg_source=full_agg_source,
            train_ref=full_train_feat,
            feature_cols=feature_cols,
            models=final_models,
            model_types=final_model_types,
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
            models=models,
            model_types=model_types,
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
        "batch_scores": batch_scores,
        "recursive_scores": recursive_scores,
        "lgb_rmsle": min(score for name, score in batch_scores.items() if name.startswith("lgb_")),
        "xgb_rmsle": xgb_score,
        "weights": weights,
        "feature_cols": feature_cols,
        "lgb_cat_cols": lgb_cat_cols,
        "xgb_cat_cols": xgb_cat_cols,
    }
