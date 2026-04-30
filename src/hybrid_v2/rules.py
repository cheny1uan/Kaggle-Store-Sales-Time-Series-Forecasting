from __future__ import annotations

import numpy as np
import pandas as pd


def build_zero_lookup(train_df: pd.DataFrame) -> pd.MultiIndex:
    zero_pairs = train_df.groupby(["store_nbr", "family"])["sales"].sum().reset_index()
    zero_pairs = zero_pairs[zero_pairs["sales"] == 0][["store_nbr", "family"]]
    return pd.MultiIndex.from_frame(zero_pairs)


def apply_zero_rule(pred_df: pd.DataFrame, zero_lookup: pd.MultiIndex) -> np.ndarray:
    out = pred_df.copy()
    index = pd.MultiIndex.from_frame(out[["store_nbr", "family"]])
    mask = index.isin(zero_lookup)
    if "onpromotion" in out.columns:
        mask = mask & (out["onpromotion"].fillna(0).astype(float) <= 0)
    preds = out["pred"].to_numpy(copy=True)
    preds[mask] = 0.0
    return preds

