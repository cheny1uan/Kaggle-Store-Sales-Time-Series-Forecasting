from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.metrics import rmsle


def prepare_training_data(df: pd.DataFrame, target_col: str = "sales") -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    y = out[target_col].astype(float).values
    X = out.drop(columns=[target_col])
    return X, pd.Series(y, index=out.index, name=target_col)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {"date", "sales", "id"}
    return [c for c in df.columns if c not in drop_cols]


def encode_categories(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
):
    cat_cols = []
    for col in feature_cols:
        if train_df[col].dtype == "object":
            cat_cols.append(col)
        elif str(train_df[col].dtype).startswith("category"):
            cat_cols.append(col)
    for col in cat_cols:
        categories = pd.Index(pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).astype("string").fillna("NA").unique())
        for df in (train_df, valid_df, test_df):
            df[col] = df[col].astype("string").fillna("NA")
            df[col] = pd.Categorical(df[col], categories=categories)
    return cat_cols


def align_categories(dfs: list[pd.DataFrame], feature_cols: list[str]) -> list[str]:
    cat_cols = []
    for col in feature_cols:
        if dfs[0][col].dtype == "object" or str(dfs[0][col].dtype).startswith("category"):
            cat_cols.append(col)
    for col in cat_cols:
        categories = pd.Index(
            pd.concat([df[col] for df in dfs], axis=0)
            .astype("string")
            .fillna("NA")
            .unique()
        )
        for df in dfs:
            df[col] = pd.Categorical(df[col].astype("string").fillna("NA"), categories=categories)
    return cat_cols


def _categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    cat_cols = []
    for col in feature_cols:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            cat_cols.append(col)
    return cat_cols


def prepare_lgbm_frames(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    test_stub = X_valid.copy()
    cat_cols = encode_categories(X_train, X_valid, test_stub, feature_cols)
    return X_train, X_valid, cat_cols


def train_lgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "sales",
    n_estimators: int = 2000,
    learning_rate: float = 0.03,
    num_leaves: int = 128,
    min_child_samples: int = 50,
    early_stopping_rounds: int = 100,
):
    X_train, X_valid, cat_cols = prepare_lgbm_frames(train_df, valid_df, feature_cols)
    y_train = np.log1p(train_df[target_col].astype(float).values)
    y_valid = valid_df[target_col].astype(float).values

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, np.log1p(y_valid))],
        eval_metric="rmse",
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    valid_pred = np.expm1(model.predict(X_valid, num_iteration=model.best_iteration_))
    score = rmsle(valid_df[target_col].values, valid_pred)
    return model, score, cat_cols


def fit_final_lgbm(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    n_estimators: int,
):
    X_train = train_df[feature_cols].copy()
    cat_cols = _categorical_columns(train_df, feature_cols)
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=128,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, np.log1p(train_df["sales"].astype(float).values), categorical_feature=cat_cols if cat_cols else "auto")
    return model, cat_cols


def predict(model, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    preds = np.expm1(model.predict(df[feature_cols], num_iteration=model.best_iteration_))
    return np.clip(preds, 0, None)
