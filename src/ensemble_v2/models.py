from __future__ import annotations

import importlib.util

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.models.train_lgbm import align_categories, prepare_lgbm_frames
from src.utils.metrics import rmsle


def xgboost_available() -> bool:
    return importlib.util.find_spec("xgboost") is not None


def _categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        c
        for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]


def prepare_xgb_frames(dfs: list[pd.DataFrame], feature_cols: list[str]) -> tuple[list[pd.DataFrame], list[str]]:
    out = [df[feature_cols].copy() for df in dfs]
    cat_cols = _categorical_columns(out[0], feature_cols)
    for col in cat_cols:
        categories = pd.Index(
            pd.concat([df[col] for df in out], axis=0)
            .astype("string")
            .fillna("NA")
            .unique()
        )
        for df in out:
            df[col] = pd.Categorical(df[col].astype("string").fillna("NA"), categories=categories).codes.astype("int16")
    for df in out:
        df.replace([np.inf, -np.inf], 0, inplace=True)
        df.fillna(0, inplace=True)
    return out, cat_cols


def fit_xgb_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int,
    early_stopping_rounds: int,
    seed: int = 43,
):
    (X_train, X_valid), cat_cols = prepare_xgb_frames([train_df, valid_df], feature_cols)
    y_train = np.log1p(train_df["sales"].astype(float).values)
    y_valid_log = np.log1p(valid_df["sales"].astype(float).values)

    if xgboost_available():
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.2,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid_log)],
                verbose=False,
                early_stopping_rounds=early_stopping_rounds,
            )
        except TypeError:
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid_log)], verbose=False)
        valid_pred = np.expm1(model.predict(X_valid))
        return model, rmsle(valid_df["sales"].values, valid_pred), cat_cols

    model = Ridge(alpha=2.0, random_state=seed)
    model.fit(X_train, y_train)
    valid_pred = np.expm1(model.predict(X_valid))
    return model, rmsle(valid_df["sales"].values, valid_pred), cat_cols


def fit_lgb_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int,
    early_stopping_rounds: int,
    seed: int = 42,
):
    X_train, X_valid, cat_cols = prepare_lgbm_frames(train_df, valid_df, feature_cols)
    y_train = np.log1p(train_df["sales"].astype(float).values)
    y_valid = np.log1p(valid_df["sales"].astype(float).values)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=50,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.02,
        reg_lambda=0.05,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    valid_pred = np.expm1(model.predict(X_valid, num_iteration=model.best_iteration_))
    return model, rmsle(valid_df["sales"].values, valid_pred), cat_cols


def fit_final_lgb_model(train_df: pd.DataFrame, feature_cols: list[str], n_estimators: int, seed: int = 42):
    X_train = train_df[feature_cols].copy()
    cat_cols = align_categories([X_train], feature_cols)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=50,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_alpha=0.02,
        reg_lambda=0.05,
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        n_jobs=-1,
    )
    model.fit(X_train, np.log1p(train_df["sales"].astype(float).values), categorical_feature=cat_cols if cat_cols else "auto")
    return model, cat_cols


def fit_final_xgb_model(train_df: pd.DataFrame, feature_cols: list[str], n_estimators: int, seed: int = 43):
    (X_train,), cat_cols = prepare_xgb_frames([train_df], feature_cols)
    if xgboost_available():
        import xgboost as xgb

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.2,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, np.log1p(train_df["sales"].astype(float).values), verbose=False)
        return model, cat_cols

    model = Ridge(alpha=2.0, random_state=seed)
    model.fit(X_train, np.log1p(train_df["sales"].astype(float).values))
    return model, cat_cols


def predict_lgb(model, train_ref: pd.DataFrame, pred_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X_ref = train_ref[feature_cols].copy()
    X_pred = pred_df[feature_cols].copy()
    align_categories([X_ref, X_pred], feature_cols)
    return np.clip(np.expm1(model.predict(X_pred, num_iteration=getattr(model, "best_iteration_", None))), 0, None)


def predict_xgb(model, train_ref: pd.DataFrame, pred_df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    (X_ref, X_pred), _ = prepare_xgb_frames([train_ref, pred_df], feature_cols)
    _ = X_ref
    return np.clip(np.expm1(model.predict(X_pred)), 0, None)
