from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.hybrid_v2.features import add_fourier_terms


def build_trend_matrix(df: pd.DataFrame, fourier_orders: tuple[int, ...]) -> pd.DataFrame:
    out = df[["date"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["t"] = (out["date"] - pd.Timestamp("2013-01-01")).dt.days.astype(float)
    for order in fourier_orders:
        out = add_fourier_terms(out, period=365, order=order, prefix=f"year_{order}")
        out = add_fourier_terms(out, period=7, order=min(order, 3), prefix=f"week_{order}")
    return out.drop(columns=["date"])


def fit_trend_model(train_df: pd.DataFrame, y: pd.Series, fourier_orders: tuple[int, ...], alpha: float) -> tuple[Ridge, list[str]]:
    X = build_trend_matrix(train_df, fourier_orders)
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X, np.log1p(y.astype(float).values))
    return model, list(X.columns)


def predict_trend(model: Ridge, df: pd.DataFrame, feature_cols: list[str], fourier_orders: tuple[int, ...]) -> np.ndarray:
    X = build_trend_matrix(df, fourier_orders)
    X = X.reindex(columns=feature_cols, fill_value=0)
    return np.expm1(model.predict(X))
