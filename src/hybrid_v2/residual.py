from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {"date", "sales", "id", "trend_pred"}
    return [c for c in df.columns if c not in drop_cols]


def _categorical_columns(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [
        c for c in feature_cols
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]


def _apply_category_levels(df: pd.DataFrame, category_levels: dict[str, pd.Index]) -> pd.DataFrame:
    out = df.copy()
    for col, cats in category_levels.items():
        out[col] = pd.Categorical(out[col].astype("string").fillna("NA"), categories=cats)
    return out


def fit_residual_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str], n_estimators: int, learning_rate: float, num_leaves: int, min_child_samples: int, early_stopping_rounds: int):
    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    y_train = np.log1p(train_df["sales"].astype(float).values) - np.log1p(train_df["trend_pred"].astype(float).values)
    y_valid = np.log1p(valid_df["sales"].astype(float).values) - np.log1p(valid_df["trend_pred"].astype(float).values)

    cat_cols = _categorical_columns(X_train, feature_cols)
    category_levels: dict[str, pd.Index] = {}
    for col in cat_cols:
        cats = pd.Index(pd.concat([X_train[col], X_valid[col]], axis=0).astype("string").fillna("NA").unique())
        category_levels[col] = cats
    X_train = _apply_category_levels(X_train, category_levels)
    X_valid = _apply_category_levels(X_valid, category_levels)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
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
    return model, cat_cols, category_levels


def predict_residuals(model, df: pd.DataFrame, feature_cols: list[str], category_levels: dict[str, pd.Index] | None = None) -> np.ndarray:
    X = df[feature_cols].copy()
    if category_levels:
        X = _apply_category_levels(X, category_levels)
    return model.predict(X, num_iteration=model.best_iteration_)
