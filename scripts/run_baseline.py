from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.load_data import load_raw_data, merge_features
from src.features.make_features import (
    add_date_features,
    add_lag_rolling_features,
    apply_target_aggregates,
    build_target_aggregates,
)
from src.models.train_lgbm import align_categories, fit_final_lgbm, get_feature_columns, train_lgbm


DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SUB_DIR = ROOT / "submissions"


MODES = {
    "fast": {
        "history_days": 180,
        "valid_days": 7,
        "n_estimators": 800,
        "early_stopping": 50,
        "full_retrain": False,
        "test_history": "train_part",
        "output_name": "baseline_fast.csv",
    },
    "full": {
        "history_days": None,
        "valid_days": 28,
        "n_estimators": 4000,
        "early_stopping": 100,
        "full_retrain": True,
        "test_history": "train_raw",
        "output_name": "baseline_full.csv",
    },
}

FEATURE_DROP_LIST = {
    "bridge_flag",
    "workday_flag",
    "year",
    "quarter",
    "holiday_any",
    "holiday_flag",
    "is_month_end",
    "is_month_start",
    "additional_flag",
    "holiday_regional",
    "holiday_cnt",
}


def build_dataset(
    base_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    agg_source_df: pd.DataFrame,
) -> pd.DataFrame:
    base = merge_features(
        base=base_df,
        stores=frames["stores"],
        oil=frames["oil"],
        holidays=frames["holidays"],
        transactions=frames["transactions"],
    )
    base = add_date_features(base)
    agg_maps = build_target_aggregates(agg_source_df)
    base = apply_target_aggregates(base, agg_maps)
    return base


def add_time_series_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sales_history = None
    transactions_history = None
    if history_df is not None:
        sales_cols = [c for c in ["date", "store_nbr", "family", "sales"] if c in history_df.columns]
        transactions_cols = [c for c in ["date", "store_nbr", "transactions"] if c in history_df.columns]
        sales_history = history_df[sales_cols].copy() if sales_cols else None
        transactions_history = history_df[transactions_cols].copy() if transactions_cols else None

    out = add_lag_rolling_features(
        target_df,
        history_df=sales_history,
        group_cols=["store_nbr", "family"],
        value_col="sales",
        feature_prefix="sales",
    )
    out = add_lag_rolling_features(
        out,
        history_df=transactions_history,
        group_cols=["store_nbr"],
        value_col="transactions",
        feature_prefix="transactions",
    )
    return out


def prepare_time_split(train_raw: pd.DataFrame, mode_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_raw = train_raw.copy()
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    max_date = train_raw["date"].max()
    valid_days = mode_cfg["valid_days"]
    split_date = max_date - pd.Timedelta(days=valid_days)
    valid_part = train_raw[train_raw["date"] > split_date].copy()
    train_part = train_raw[train_raw["date"] <= split_date].copy()

    history_days = mode_cfg["history_days"]
    if history_days is not None:
        history_start = split_date - pd.Timedelta(days=history_days)
        train_part = train_part[train_part["date"] > history_start].copy()

    return train_part, valid_part, split_date


def align_feature_frames(
    train_feat: pd.DataFrame,
    valid_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    all_feature_cols = get_feature_columns(train_feat)
    feature_cols = [c for c in all_feature_cols if c not in FEATURE_DROP_LIST]
    removed_cols = [c for c in all_feature_cols if c in FEATURE_DROP_LIST]
    train_cols = ["date", "sales"] + feature_cols
    valid_cols = ["date", "sales"] + feature_cols
    test_cols = ["date", "id"] + feature_cols

    train_feat = train_feat.reindex(columns=train_cols, fill_value=0)
    valid_feat = valid_feat.reindex(columns=valid_cols, fill_value=0)
    test_feat = test_feat.reindex(columns=test_cols, fill_value=0)
    return train_feat, valid_feat, test_feat, feature_cols, removed_cols


def save_feature_importance(model, feature_cols: list[str], out_path: Path) -> pd.DataFrame:
    booster = model.booster_ if hasattr(model, "booster_") else model
    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False)
    fi.to_csv(out_path, index=False)
    print("\nTop 30 feature importance:")
    print(fi.head(30).to_string(index=False))
    print(f"Saved feature importance to {out_path}")
    return fi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast: small window for debugging, full: full training and final retrain",
    )
    args = parser.parse_args()
    cfg = MODES[args.mode]

    OUTPUT_DIR.mkdir(exist_ok=True)
    SUB_DIR.mkdir(exist_ok=True)

    frames = load_raw_data(DATA_DIR)
    train_raw = frames["train"].copy()
    test_raw = frames["test"].copy()
    train_raw["date"] = pd.to_datetime(train_raw["date"])
    test_raw["date"] = pd.to_datetime(test_raw["date"])

    train_part, valid_part, split_date = prepare_time_split(train_raw, cfg)
    print(f"Mode: {args.mode}")
    print(f"Train rows: {len(train_part):,}")
    print(f"Valid rows: {len(valid_part):,}")
    print(f"Split date: {split_date.date()}")

    agg_source = add_date_features(train_part if args.mode == "fast" else train_raw)
    train_feat = build_dataset(train_part, frames, agg_source)
    valid_feat = build_dataset(valid_part, frames, agg_source)
    test_feat = build_dataset(test_raw, frames, agg_source)
    train_feat = add_time_series_features(train_feat)
    valid_feat = add_time_series_features(valid_feat, history_df=train_feat)
    test_history = train_feat[["date", "store_nbr", "family", "sales", "transactions"]].copy()
    test_feat = add_time_series_features(test_feat, history_df=test_history)
    train_feat, valid_feat, test_feat, feature_cols, removed = align_feature_frames(train_feat, valid_feat, test_feat)
    print(f"Dropped low-importance features: {removed}")

    model, val_score, cat_cols = train_lgbm(
        train_feat,
        valid_feat,
        feature_cols,
        n_estimators=cfg["n_estimators"],
        early_stopping_rounds=cfg["early_stopping"],
    )
    print(f"Validation RMSLE: {val_score:.5f}")
    print(f"Categorical features: {cat_cols}")
    save_feature_importance(model, feature_cols, OUTPUT_DIR / f"feature_importance_{args.mode}.csv")

    if cfg["full_retrain"]:
        full_agg_source = add_date_features(train_raw)
        full_train_feat = build_dataset(train_raw, frames, full_agg_source)
        full_train_feat = add_time_series_features(full_train_feat)
        full_train_feat = full_train_feat.reindex(columns=["date", "sales"] + feature_cols, fill_value=0)
        final_test_feat = build_dataset(test_raw, frames, full_agg_source)
        final_test_feat = add_time_series_features(final_test_feat, history_df=full_train_feat[["date", "store_nbr", "family", "sales", "transactions"]])
        final_test_feat = final_test_feat.reindex(columns=["date", "id"] + feature_cols, fill_value=0)

        X_train = full_train_feat[feature_cols].copy()
        X_test = final_test_feat[feature_cols].copy()
        cat_cols = align_categories([X_train, X_test], feature_cols)
        full_train_feat[feature_cols] = X_train
        final_test_feat[feature_cols] = X_test

        final_model, _ = fit_final_lgbm(full_train_feat, feature_cols, n_estimators=model.best_iteration_ or cfg["n_estimators"])
        save_feature_importance(final_model, feature_cols, OUTPUT_DIR / "feature_importance_full_final.csv")
        preds = np.expm1(final_model.predict(final_test_feat[feature_cols]))
    else:
        X_test = test_feat[feature_cols].copy()
        X_train = train_feat[feature_cols].copy()
        cat_cols = align_categories([X_train, X_test], feature_cols)
        train_feat[feature_cols] = X_train
        test_feat[feature_cols] = X_test
        preds = np.expm1(model.predict(test_feat[feature_cols], num_iteration=model.best_iteration_))

    preds = pd.Series(preds).clip(lower=0)
    submission = frames["sample_submission"].copy()
    submission["sales"] = preds.values
    out_path = SUB_DIR / cfg["output_name"]
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
