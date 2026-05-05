from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_raw_data, merge_features
from src.ensemble_v3.config import EnsembleV3Config, MODES
from src.ensemble_v3.ensemble import blend_predictions
from src.ensemble_v3.features import (
    ZERO_KEY_COLS,
    add_sales_time_series_features,
    add_localized_holiday_features,
    add_transaction_time_series_features,
    align_feature_frames,
    apply_transaction_aggregates,
    apply_zero_forecast,
    build_sales_dataset,
    build_transaction_aggregates,
    build_transaction_base_features,
    build_recent_zero_set,
    build_zero_set,
    prepare_time_split,
)
from src.ensemble_v2_plus.models import (
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


def _store_date_base(base_df: pd.DataFrame) -> pd.DataFrame:
    out = base_df[["date", "store_nbr"]].drop_duplicates().copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values(["date", "store_nbr"]).reset_index(drop=True)


def _history_window(df: pd.DataFrame, anchor_date: pd.Timestamp, history_days: int | None) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if history_days is None:
        return out
    start_date = pd.Timestamp(anchor_date) - pd.Timedelta(days=history_days)
    return out[out["date"] > start_date].copy()


def _build_sales_feature_sets(
    frames: dict[str, pd.DataFrame],
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    test_raw: pd.DataFrame,
    agg_source: pd.DataFrame,
    feature_drop: set[str],
    valid_tx_override: pd.DataFrame | None = None,
    test_tx_override: pd.DataFrame | None = None,
    use_long_lags: bool = False,
    use_reference_features: bool = False,
):
    train_feat = build_sales_dataset(train_part, frames, agg_source)
    valid_feat = build_sales_dataset(valid_part, frames, agg_source, transactions_override=valid_tx_override)
    test_feat = build_sales_dataset(test_raw, frames, agg_source, transactions_override=test_tx_override)
    train_feat = add_sales_time_series_features(
        train_feat,
        use_long_lags=use_long_lags,
        use_reference_features=use_reference_features,
    )
    valid_feat = add_sales_time_series_features(
        valid_feat,
        history_df=train_feat,
        use_long_lags=use_long_lags,
        use_reference_features=use_reference_features,
    )
    test_history_cols = [c for c in ["date", "store_nbr", "family", "sales", "transactions", "onpromotion"] if c in train_feat.columns]
    test_history = train_feat[test_history_cols].copy()
    test_feat = add_sales_time_series_features(
        test_feat,
        history_df=test_history,
        use_long_lags=use_long_lags,
        use_reference_features=use_reference_features,
    )
    return align_feature_frames(train_feat, valid_feat, test_feat, feature_drop)


def _build_transaction_feature_sets(
    frames: dict[str, pd.DataFrame],
    train_tx: pd.DataFrame,
    valid_tx: pd.DataFrame,
    test_tx: pd.DataFrame,
    feature_drop: set[str],
):
    train_base = build_transaction_base_features(train_tx, frames)
    agg_maps = build_transaction_aggregates(train_base)
    train_feat = apply_transaction_aggregates(train_base, agg_maps)
    valid_base = build_transaction_base_features(valid_tx, frames)
    valid_feat = apply_transaction_aggregates(valid_base, agg_maps)
    test_base = build_transaction_base_features(test_tx, frames)
    test_feat = apply_transaction_aggregates(test_base, agg_maps)

    train_feat = add_transaction_time_series_features(train_feat)
    valid_feat = add_transaction_time_series_features(valid_feat, history_df=train_feat)
    test_feat = add_transaction_time_series_features(test_feat, history_df=train_feat)

    train_fit = train_feat.rename(columns={"transactions": "sales"})
    valid_fit = valid_feat.rename(columns={"transactions": "sales"})
    test_fit = test_feat.rename(columns={"transactions": "sales"})
    return align_feature_frames(train_fit, valid_fit, test_fit, feature_drop), agg_maps


def recursive_transactions_forecast(
    base_df: pd.DataFrame,
    history_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    agg_maps: dict[str, pd.DataFrame],
    train_ref: pd.DataFrame,
    feature_cols: list[str],
    models: dict[str, object],
    model_types: dict[str, str],
    weights: dict[str, float],
) -> pd.DataFrame:
    base = _store_date_base(base_df)
    history = history_df[["date", "store_nbr", "transactions"]].copy()
    history["date"] = pd.to_datetime(history["date"])
    frames_out: list[pd.DataFrame] = []
    max_lookback = 70

    for current_date in sorted(pd.to_datetime(base["date"]).unique()):
        day_base = base[base["date"] == current_date].copy()
        day_base["transactions"] = 0.0
        recent_history = history[history["date"] >= pd.Timestamp(current_date) - pd.Timedelta(days=max_lookback)].copy()
        day_feat = build_transaction_base_features(day_base, frames)
        day_feat = apply_transaction_aggregates(day_feat, agg_maps)
        day_feat = add_transaction_time_series_features(day_feat, history_df=recent_history)
        day_feat = day_feat.reindex(columns=["date"] + feature_cols, fill_value=0)

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
        day_out = day_base[["date", "store_nbr"]].copy()
        day_out["transactions"] = day_pred
        frames_out.append(day_out)

        history = pd.concat([history, day_out], ignore_index=True, sort=False)

    return pd.concat(frames_out, ignore_index=True, sort=False)


def recursive_sales_forecast(
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
    tx_pred_df: pd.DataFrame,
    use_long_lags: bool = False,
    use_reference_features: bool = False,
) -> pd.DataFrame:
    base = build_sales_dataset(base_df, frames, agg_source, transactions_override=tx_pred_df)
    history_cols = [c for c in ["date", "store_nbr", "family", "sales", "transactions", "onpromotion"] if c in history_df.columns]
    history = history_df[history_cols].copy()
    history["date"] = pd.to_datetime(history["date"])
    frames_out: list[pd.DataFrame] = []
    max_lookback = 1120 if use_reference_features else (760 if use_long_lags else 70)

    for current_date in sorted(pd.to_datetime(base["date"]).unique()):
        day_base = base[base["date"] == current_date].copy()
        recent_history = history[history["date"] >= pd.Timestamp(current_date) - pd.Timedelta(days=max_lookback)].copy()
        day_feat = add_sales_time_series_features(
            day_base,
            history_df=recent_history,
            use_long_lags=use_long_lags,
            use_reference_features=use_reference_features,
        )
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

        day_out = day_feat[["id", "date", "store_nbr", "family", "onpromotion", "transactions"]].copy()
        day_out["pred"] = day_pred
        frames_out.append(day_out)

        history_update = day_out[["date", "store_nbr", "family", "transactions", "onpromotion"]].copy()
        history_update["sales"] = day_pred
        history = pd.concat([history, history_update], ignore_index=True, sort=False)

    return pd.concat(frames_out, ignore_index=True, sort=False)


def _build_calibration_frame(valid_part: pd.DataFrame, valid_pred_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["id", "date", "store_nbr", "family", "onpromotion", "pred"]
    out = valid_part[["id", "sales"]].merge(valid_pred_df[cols], on="id", how="left")
    out["date"] = pd.to_datetime(out["date"])
    out["pred"] = out["pred"].fillna(0).clip(lower=0)
    return out


def _strategy_specs(strategy: str) -> list[tuple[str, list[str], float]]:
    specs = {
        "none": [],
        "global": [("global", [], 3000.0)],
        "family": [("family", ["family"], 500.0)],
        "store": [("store", ["store_nbr"], 800.0)],
        "store_family": [("store_family", ["store_nbr", "family"], 80.0)],
        "family_store": [
            ("family", ["family"], 700.0),
            ("store", ["store_nbr"], 1000.0),
        ],
    }
    return specs[strategy]


def _fit_calibration(df: pd.DataFrame, strategy: str) -> dict[str, object]:
    calibration: dict[str, object] = {"strategy": strategy, "groups": [], "global": 0.0}
    if strategy == "none" or df.empty:
        return calibration

    work = df.copy()
    work["log_resid"] = np.log1p(work["sales"].clip(lower=0)) - np.log1p(work["pred"].clip(lower=0))

    for name, group_cols, prior_strength in _strategy_specs(strategy):
        if name == "global":
            shrink = len(work) / (len(work) + prior_strength)
            calibration["global"] = float(np.clip(work["log_resid"].mean() * shrink, -0.18, 0.18))
            continue

        mapping = (
            work.groupby(group_cols, observed=True)["log_resid"]
            .agg(["mean", "count"])
            .reset_index()
        )
        shrink = mapping["count"] / (mapping["count"] + prior_strength)
        mapping[f"{name}_corr"] = np.clip(mapping["mean"] * shrink, -0.18, 0.18).astype("float32")
        calibration["groups"].append(
            {
                "name": name,
                "cols": group_cols,
                "mapping": mapping[group_cols + [f"{name}_corr"]],
            }
        )
    return calibration


def _apply_calibration(df: pd.DataFrame, calibration: dict[str, object]) -> np.ndarray:
    pred = df["pred"].fillna(0).clip(lower=0).astype(float).to_numpy()
    if calibration["strategy"] == "none":
        return pred

    work = df.copy()
    total_corr = np.full(len(work), float(calibration.get("global", 0.0)), dtype=float)
    for group in calibration["groups"]:
        name = group["name"]
        cols = group["cols"]
        corr_col = f"{name}_corr"
        work = work.merge(group["mapping"], on=cols, how="left")
        total_corr += work[corr_col].fillna(0).astype(float).to_numpy()
        work.drop(columns=[corr_col], inplace=True)

    total_corr = np.clip(total_corr, -0.22, 0.22)
    calibrated = np.expm1(np.log1p(pred) + total_corr)
    calibrated[pred <= 0] = 0.0
    return np.clip(calibrated, 0, None)


def _blend_logspace(raw_pred: np.ndarray, adjusted_pred: np.ndarray, alpha: float) -> np.ndarray:
    raw = np.asarray(raw_pred, dtype=float)
    adjusted = np.asarray(adjusted_pred, dtype=float)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = np.expm1((1.0 - alpha) * np.log1p(np.clip(raw, 0, None)) + alpha * np.log1p(np.clip(adjusted, 0, None)))
    blended[raw <= 0] = 0.0
    return np.clip(blended, 0, None)


def _search_blend_weights(pred_map: dict[str, np.ndarray], y_true: np.ndarray, step: float) -> tuple[dict[str, float], float]:
    names = list(pred_map)
    if not names:
        return {}, float("inf")
    if len(names) == 1:
        pred = blend_predictions(pred_map, {names[0]: 1.0})
        return {names[0]: 1.0}, rmsle(y_true, pred)

    grid = np.linspace(0.0, 1.0, int(round(1.0 / step)) + 1)
    best_weights = {name: 1.0 / len(names) for name in names}
    best_score = float("inf")

    if len(names) == 2:
        for w0 in grid:
            weights = {names[0]: float(w0), names[1]: float(1.0 - w0)}
            pred = blend_predictions(pred_map, weights)
            score = rmsle(y_true, pred)
            if score < best_score:
                best_score = score
                best_weights = weights
    elif len(names) == 3:
        for w0 in grid:
            for w1 in grid:
                w2 = 1.0 - w0 - w1
                if w2 < 0:
                    continue
                weights = {names[0]: float(w0), names[1]: float(w1), names[2]: float(w2)}
                pred = blend_predictions(pred_map, weights)
                score = rmsle(y_true, pred)
                if score < best_score:
                    best_score = score
                    best_weights = weights
    else:
        # Fall back to the score-based heuristic for larger blends.
        from src.ensemble_v3.ensemble import inverse_score_weights

        best_weights = inverse_score_weights({name: rmsle(y_true, pred_map[name]) for name in names})
        best_score = rmsle(y_true, blend_predictions(pred_map, best_weights))

    return best_weights, best_score


def _select_postprocess(
    valid_part: pd.DataFrame,
    valid_pred_df: pd.DataFrame,
    valid_days: int,
    frames: dict[str, pd.DataFrame],
    zero_set,
) -> tuple[dict[str, object], pd.DataFrame, float]:
    frame = _build_calibration_frame(valid_part, valid_pred_df)
    eval_days = 2 if valid_days <= 7 else 7
    split_date = frame["date"].max() - pd.Timedelta(days=eval_days)
    calib_train = frame[frame["date"] <= split_date].copy()
    calib_eval = frame[frame["date"] > split_date].copy()

    report_rows = []
    base_score = rmsle(calib_eval["sales"].values, calib_eval["pred"].values)
    report_rows.append({"strategy": "none", "alpha": 0.0, "calibration_holdout_rmsle": base_score})

    best_strategy = "none"
    best_alpha = 0.0
    best_score = base_score
    for strategy in ["global", "family", "store", "store_family", "family_store"]:
        calibration = _fit_calibration(calib_train, strategy)
        cal_pred = _apply_calibration(calib_eval, calibration)
        for alpha in np.linspace(0.0, 1.0, 21):
            pred = _blend_logspace(calib_eval["pred"].values, cal_pred, alpha)
            score = rmsle(calib_eval["sales"].values, pred)
            report_rows.append({"strategy": strategy, "alpha": float(alpha), "calibration_holdout_rmsle": score})
            if score < best_score - 1e-5:
                best_strategy = strategy
                best_alpha = float(alpha)
                best_score = score

    final_calibration = _fit_calibration(frame, best_strategy)
    final_calibrated = _apply_calibration(frame, final_calibration)
    final_pred = _blend_logspace(frame["pred"].values, final_calibrated, best_alpha)
    final_score = rmsle(frame["sales"].values, final_pred)
    report = pd.DataFrame(report_rows).sort_values("calibration_holdout_rmsle")
    return {
        "kind": "calibration",
        "calibration": final_calibration,
        "alpha": best_alpha,
        "report": report,
        "score": final_score,
    }


def run(mode: str = "fast") -> dict[str, object]:
    config = EnsembleV3Config(mode=mode)
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
    tx_raw = frames["transactions"].copy()
    tx_raw["date"] = pd.to_datetime(tx_raw["date"])
    tx_train_part, tx_valid_part, _ = prepare_time_split(tx_raw, mode_cfg.valid_days, mode_cfg.history_days)

    sales_agg_source = add_date_features(
        merge_features(
            base=train_part,
            stores=frames["stores"],
            oil=frames["oil"],
            holidays=frames["holidays"],
            transactions=frames["transactions"],
        )
    )

    print(f"Mode: {mode}")
    print(f"Sales train rows: {len(train_part):,}")
    print(f"Sales valid rows: {len(valid_part):,}")
    print(f"Split date: {split_date.date()}")

    tx_feature_bundle, tx_agg_maps = _build_transaction_feature_sets(
        frames,
        tx_train_part,
        tx_valid_part,
        _store_date_base(test_raw).assign(transactions=0.0),
        config.feature_drop,
    )
    tx_train_feat, tx_valid_feat, tx_test_feat, tx_feature_cols, tx_removed_cols = tx_feature_bundle
    print(f"Transaction features dropped: {tx_removed_cols}")
    print(f"XGBoost installed: {xgboost_available()}")

    tx_models: dict[str, object] = {}
    tx_model_types: dict[str, str] = {}
    tx_batch_scores: dict[str, float] = {}
    tx_best_iterations: dict[str, int] = {}

    for seed in mode_cfg.tx_lgb_seeds:
        name = f"tx_lgb_{seed}"
        model, score, _ = fit_lgb_model(
            tx_train_feat,
            tx_valid_feat,
            tx_feature_cols,
            n_estimators=mode_cfg.tx_estimators,
            early_stopping_rounds=mode_cfg.early_stopping,
            seed=seed,
        )
        tx_models[name] = model
        tx_model_types[name] = "lgb"
        tx_batch_scores[name] = score
        tx_best_iterations[name] = getattr(model, "best_iteration_", None) or mode_cfg.tx_estimators
        print(f"Transaction LightGBM seed {seed} validation RMSLE: {score:.5f}")

    tx_xgb_model, tx_xgb_score, _ = fit_xgb_model(
        tx_train_feat,
        tx_valid_feat,
        tx_feature_cols,
        n_estimators=mode_cfg.tx_xgb_estimators,
        early_stopping_rounds=mode_cfg.early_stopping,
        seed=mode_cfg.tx_xgb_seed,
    )
    tx_xgb_name = f"tx_xgb_{mode_cfg.tx_xgb_seed}"
    tx_models[tx_xgb_name] = tx_xgb_model
    tx_model_types[tx_xgb_name] = "xgb"
    tx_batch_scores[tx_xgb_name] = tx_xgb_score
    print(f"Transaction XGBoost seed {mode_cfg.tx_xgb_seed} validation RMSLE: {tx_xgb_score:.5f}")

    tx_recursive_scores: dict[str, float] = {}
    tx_recursive_pred_dfs: dict[str, pd.DataFrame] = {}
    for name in tx_models:
        one_hot = {model_name: 0.0 for model_name in tx_models}
        one_hot[name] = 1.0
        pred_df = recursive_transactions_forecast(
            base_df=tx_valid_part,
            history_df=tx_train_part,
            frames=frames,
            agg_maps=tx_agg_maps,
            train_ref=tx_train_feat,
            feature_cols=tx_feature_cols,
            models=tx_models,
            model_types=tx_model_types,
            weights=one_hot,
        )
        tx_recursive_pred_dfs[name] = pred_df
        eval_df = tx_valid_part[["date", "store_nbr", "transactions"]].merge(
            pred_df[["date", "store_nbr", "transactions"]],
            on=["date", "store_nbr"],
            how="left",
        )
        tx_recursive_scores[name] = rmsle(eval_df["transactions_x"].values, eval_df["transactions_y"].fillna(0).values)
        print(f"Recursive transaction {name} RMSLE: {tx_recursive_scores[name]:.5f}")

    tx_eval_pred_map = {}
    tx_eval_true = None
    for name, pred_df in tx_recursive_pred_dfs.items():
        eval_df = tx_valid_part[["date", "store_nbr", "transactions"]].merge(
            pred_df[["date", "store_nbr", "transactions"]],
            on=["date", "store_nbr"],
            how="left",
        ).sort_values(["date", "store_nbr"])
        tx_eval_pred_map[name] = eval_df["transactions_y"].fillna(0).to_numpy()
        if tx_eval_true is None:
            tx_eval_true = eval_df["transactions_x"].to_numpy()
    tx_weights, tx_blend_score = _search_blend_weights(tx_eval_pred_map, tx_eval_true, step=0.02)
    print(f"Transaction blend weights: {tx_weights}")
    print(f"Recursive transaction ensemble validation RMSLE: {tx_blend_score:.5f}")

    tx_valid_pred_df = recursive_transactions_forecast(
        base_df=tx_valid_part,
        history_df=tx_train_part,
        frames=frames,
        agg_maps=tx_agg_maps,
        train_ref=tx_train_feat,
        feature_cols=tx_feature_cols,
        models=tx_models,
        model_types=tx_model_types,
        weights=tx_weights,
    )
    tx_valid_eval = tx_valid_part[["date", "store_nbr", "transactions"]].merge(
        tx_valid_pred_df[["date", "store_nbr", "transactions"]],
        on=["date", "store_nbr"],
        how="left",
    )
    tx_blend_score = rmsle(tx_valid_eval["transactions_x"].values, tx_valid_eval["transactions_y"].fillna(0).values)
    print(f"Final transaction validation RMSLE: {tx_blend_score:.5f}")

    sales_train_feat, sales_valid_feat, sales_test_feat, sales_feature_cols, sales_removed_cols = _build_sales_feature_sets(
        frames,
        train_part,
        valid_part,
        test_raw,
        sales_agg_source,
        config.feature_drop,
        valid_tx_override=tx_valid_pred_df,
        use_long_lags=mode_cfg.use_long_lags,
        use_reference_features=mode_cfg.use_reference_features,
    )
    print(f"Sales features dropped: {sales_removed_cols}")

    sales_models: dict[str, object] = {}
    sales_model_types: dict[str, str] = {}
    sales_batch_scores: dict[str, float] = {}
    sales_best_iterations: dict[str, int] = {}

    for seed in mode_cfg.lgb_seeds:
        name = f"lgb_{seed}"
        model, score, _ = fit_lgb_model(
            sales_train_feat,
            sales_valid_feat,
            sales_feature_cols,
            n_estimators=mode_cfg.sales_estimators,
            early_stopping_rounds=mode_cfg.early_stopping,
            seed=seed,
        )
        sales_models[name] = model
        sales_model_types[name] = "lgb"
        sales_batch_scores[name] = score
        sales_best_iterations[name] = getattr(model, "best_iteration_", None) or mode_cfg.sales_estimators
        print(f"Sales LightGBM seed {seed} validation RMSLE: {score:.5f}")

    sales_xgb_model, sales_xgb_score, _ = fit_xgb_model(
        sales_train_feat,
        sales_valid_feat,
        sales_feature_cols,
        n_estimators=mode_cfg.sales_xgb_estimators,
        early_stopping_rounds=mode_cfg.early_stopping,
        seed=mode_cfg.xgb_seed,
    )
    sales_xgb_name = f"xgb_{mode_cfg.xgb_seed}"
    sales_models[sales_xgb_name] = sales_xgb_model
    sales_model_types[sales_xgb_name] = "xgb"
    sales_batch_scores[sales_xgb_name] = sales_xgb_score
    print(f"Sales XGBoost seed {mode_cfg.xgb_seed} validation RMSLE: {sales_xgb_score:.5f}")

    zero_set = build_zero_set(train_part)
    if mode_cfg.zero_recent_days is not None:
        zero_set = zero_set.union(build_recent_zero_set(train_part, mode_cfg.zero_recent_days))
    print(f"Zero-sales store-family pairs: {len(zero_set):,}")

    sales_recursive_scores: dict[str, float] = {}
    sales_recursive_pred_dfs: dict[str, pd.DataFrame] = {}
    for name in sales_models:
        one_hot = {model_name: 0.0 for model_name in sales_models}
        one_hot[name] = 1.0
        pred_df = recursive_sales_forecast(
            base_df=valid_part,
            history_df=sales_train_feat,
            frames=frames,
            agg_source=sales_agg_source,
            train_ref=sales_train_feat,
            feature_cols=sales_feature_cols,
            models=sales_models,
            model_types=sales_model_types,
            weights=one_hot,
            zero_set=zero_set,
            tx_pred_df=tx_valid_pred_df,
            use_long_lags=mode_cfg.use_long_lags,
            use_reference_features=mode_cfg.use_reference_features,
        )
        sales_recursive_pred_dfs[name] = pred_df
        eval_df = valid_part[["id", "sales"]].merge(pred_df[["id", "pred"]], on="id", how="left")
        sales_recursive_scores[name] = rmsle(eval_df["sales"].values, eval_df["pred"].fillna(0).values)
        print(f"Recursive sales {name} RMSLE: {sales_recursive_scores[name]:.5f}")

    sales_eval_pred_map = {}
    sales_eval_true = None
    for name, pred_df in sales_recursive_pred_dfs.items():
        eval_df = valid_part[["id", "sales"]].merge(pred_df[["id", "pred"]], on="id", how="left").sort_values("id")
        sales_eval_pred_map[name] = eval_df["pred"].fillna(0).to_numpy()
        if sales_eval_true is None:
            sales_eval_true = eval_df["sales"].to_numpy()
    sales_weights, sales_blend_score = _search_blend_weights(sales_eval_pred_map, sales_eval_true, step=0.05)
    print(f"Sales blend weights: {sales_weights}")

    valid_pred_df = recursive_sales_forecast(
        base_df=valid_part,
        history_df=sales_train_feat,
        frames=frames,
        agg_source=sales_agg_source,
        train_ref=sales_train_feat,
        feature_cols=sales_feature_cols,
        models=sales_models,
        model_types=sales_model_types,
        weights=sales_weights,
        zero_set=zero_set,
        tx_pred_df=tx_valid_pred_df,
        use_long_lags=mode_cfg.use_long_lags,
        use_reference_features=mode_cfg.use_reference_features,
    )
    valid_eval = valid_part[["id", "sales"]].merge(valid_pred_df[["id", "pred"]], on="id", how="left")
    blend_score = rmsle(valid_eval["sales"].values, valid_eval["pred"].fillna(0).values)
    print(f"Recursive sales ensemble validation RMSLE: {blend_score:.5f}")

    postprocess = _select_postprocess(valid_part, valid_pred_df, mode_cfg.valid_days, frames, zero_set)
    cal_report = postprocess["report"]
    calibrated_score = postprocess["score"]
    print("Calibration report:")
    print(cal_report.to_string(index=False))
    print(f"Chosen calibration strategy: {postprocess['calibration']['strategy']}")
    print(f"Calibration blend alpha: {postprocess['alpha']:.2f}")
    print(f"Calibrated validation RMSLE: {calibrated_score:.5f}")

    if mode_cfg.full_retrain:
        final_train_raw = _history_window(train_raw, train_raw["date"].max(), mode_cfg.history_days)
        final_tx_raw = _history_window(tx_raw, train_raw["date"].max(), mode_cfg.history_days)

        full_tx_base = build_transaction_base_features(final_tx_raw, frames)
        full_tx_agg_maps = build_transaction_aggregates(full_tx_base)
        full_tx_feat = apply_transaction_aggregates(full_tx_base, full_tx_agg_maps)
        full_tx_feat = add_transaction_time_series_features(full_tx_feat)
        full_tx_fit = full_tx_feat.rename(columns={"transactions": "sales"})

        final_tx_models: dict[str, object] = {}
        final_tx_model_types: dict[str, str] = {}
        for seed in mode_cfg.tx_lgb_seeds:
            name = f"tx_lgb_{seed}"
            if tx_weights.get(name, 0.0) == 0.0:
                continue
            final_tx_models[name] = fit_final_lgb_model(
                full_tx_fit,
                tx_feature_cols,
                n_estimators=tx_best_iterations[name],
                seed=seed,
            )[0]
            final_tx_model_types[name] = "lgb"
        if tx_weights.get(tx_xgb_name, 0.0) > 0.0:
            final_tx_models[tx_xgb_name] = fit_final_xgb_model(
                full_tx_fit,
                tx_feature_cols,
                n_estimators=mode_cfg.tx_xgb_estimators,
                seed=mode_cfg.tx_xgb_seed,
            )[0]
            final_tx_model_types[tx_xgb_name] = "xgb"

        tx_test_pred_df = recursive_transactions_forecast(
            base_df=_store_date_base(test_raw).assign(transactions=0.0),
            history_df=final_tx_raw,
            frames=frames,
            agg_maps=full_tx_agg_maps,
            train_ref=full_tx_fit,
            feature_cols=tx_feature_cols,
            models=final_tx_models,
            model_types=final_tx_model_types,
            weights=tx_weights,
        )

        full_sales_agg_source = add_date_features(
            merge_features(
                base=final_train_raw,
                stores=frames["stores"],
                oil=frames["oil"],
                holidays=frames["holidays"],
                transactions=frames["transactions"],
            )
        )
        full_sales_train_feat = build_sales_dataset(final_train_raw, frames, full_sales_agg_source)
        full_sales_train_feat = add_sales_time_series_features(
            full_sales_train_feat,
            use_long_lags=mode_cfg.use_long_lags,
            use_reference_features=mode_cfg.use_reference_features,
        )
        full_sales_train_feat = full_sales_train_feat.reindex(columns=["date", "sales"] + sales_feature_cols, fill_value=0)

        final_sales_models: dict[str, object] = {}
        final_sales_model_types: dict[str, str] = {}
        for seed in mode_cfg.lgb_seeds:
            name = f"lgb_{seed}"
            if sales_weights.get(name, 0.0) == 0.0:
                continue
            final_sales_models[name] = fit_final_lgb_model(
                full_sales_train_feat,
                sales_feature_cols,
                n_estimators=sales_best_iterations[name],
                seed=seed,
            )[0]
            final_sales_model_types[name] = "lgb"
        if sales_weights.get(sales_xgb_name, 0.0) > 0.0:
            final_sales_models[sales_xgb_name] = fit_final_xgb_model(
                full_sales_train_feat,
                sales_feature_cols,
                n_estimators=mode_cfg.sales_xgb_estimators,
                seed=mode_cfg.xgb_seed,
            )[0]
            final_sales_model_types[sales_xgb_name] = "xgb"

        test_pred_df = recursive_sales_forecast(
            base_df=test_raw,
            history_df=full_sales_train_feat,
            frames=frames,
            agg_source=full_sales_agg_source,
            train_ref=full_sales_train_feat,
            feature_cols=sales_feature_cols,
            models=final_sales_models,
            model_types=final_sales_model_types,
            weights=sales_weights,
            zero_set=(
                build_zero_set(train_raw).union(build_recent_zero_set(train_raw, mode_cfg.zero_recent_days))
                if mode_cfg.zero_recent_days is not None
                else build_zero_set(train_raw)
            ),
            tx_pred_df=tx_test_pred_df,
            use_long_lags=mode_cfg.use_long_lags,
            use_reference_features=mode_cfg.use_reference_features,
        )
    else:
        test_tx_pred_df = recursive_transactions_forecast(
            base_df=_store_date_base(test_raw).assign(transactions=0.0),
            history_df=tx_train_part,
            frames=frames,
            agg_maps=tx_agg_maps,
            train_ref=tx_train_feat,
            feature_cols=tx_feature_cols,
            models=tx_models,
            model_types=tx_model_types,
            weights=tx_weights,
        )
        test_pred_df = recursive_sales_forecast(
            base_df=test_raw,
            history_df=sales_train_feat,
            frames=frames,
            agg_source=sales_agg_source,
            train_ref=sales_train_feat,
            feature_cols=sales_feature_cols,
            models=sales_models,
            model_types=sales_model_types,
            weights=sales_weights,
            zero_set=zero_set,
            tx_pred_df=test_tx_pred_df,
            use_long_lags=mode_cfg.use_long_lags,
            use_reference_features=mode_cfg.use_reference_features,
        )

    calibration = postprocess["calibration"]
    calibrated_pred = _apply_calibration(test_pred_df, calibration)
    preds = _blend_logspace(test_pred_df["pred"].values, calibrated_pred, postprocess["alpha"])
    preds = test_pred_df.assign(pred=preds).sort_values("id")["pred"].to_numpy()

    submission = frames["sample_submission"].copy()
    submission["sales"] = preds
    out_path = submission_dir / mode_cfg.output_name
    submission.to_csv(out_path, index=False)

    return {
        "submission_path": out_path,
        "validation_rmsle": calibrated_score,
        "base_validation_rmsle": blend_score,
        "tx_validation_rmsle": tx_blend_score,
        "calibration_strategy": postprocess["kind"] if postprocess["kind"] == "stacker" else postprocess["calibration"]["strategy"],
        "batch_scores": sales_batch_scores,
        "recursive_scores": sales_recursive_scores,
        "tx_batch_scores": tx_batch_scores,
        "tx_recursive_scores": tx_recursive_scores,
        "sales_weights": sales_weights,
        "tx_weights": tx_weights,
        "feature_cols": sales_feature_cols,
        "tx_feature_cols": tx_feature_cols,
    }
