from __future__ import annotations

"""构造适配 TFT baseline 的特征表。

这个模块负责从原始销售主表和辅助表中生成一份尽量精简但可直接用于
Temporal Fusion Transformer 的 baseline 特征集。这里保留的特征主要分为：
- 静态特征：门店与品类属性
- 已知未来特征：日历、促销、节假日等未来可提前知道的信息
- 历史观测特征：销量、交易量、油价等只能从历史中观察到的变量

除 baseline 主路径外，模块里也保留了聚合统计和 lag/rolling 工具，便于后续扩展。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# 静态特征：定义序列身份或门店属性，在时间上不发生变化。
STATIC_COLUMNS = ["store_nbr", "family", "city", "state", "type", "cluster"]
# 已知未来特征：未来预测期依然可以提前知道的变量。
KNOWN_FUTURE_COLUMNS = [
    "onpromotion",
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "quarter",
    "is_month_start",
    "is_month_end",
    "is_weekend",
    "is_payday",
    "days_to_next_payday",
    "days_since_prev_payday",
    "payday_window_3",
    "doy_sin",
    "doy_cos",
    "holiday_flag",
    "event_flag",
]
# 历史观测特征：训练期可观测，但进入未来预测期后不能直接提前知道。
PAST_OBSERVED_COLUMNS = ["sales", "transactions", "dcoilwtico"]
BASELINE_EXPORT_COLUMNS = ["id", "date", *STATIC_COLUMNS, *KNOWN_FUTURE_COLUMNS, *PAST_OBSERVED_COLUMNS]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """补充 TFT baseline 需要的日历类特征。

    这里既包含离散日历特征，也包含 payday 规则和年内周期的正弦/余弦编码，
    用于帮助模型学习促销节奏、月末效应和季节性变化。
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["month"] = out["date"].dt.month.astype("int8")
    out["day"] = out["date"].dt.day.astype("int8")
    out["dayofweek"] = out["date"].dt.dayofweek.astype("int8")
    dayofyear = out["date"].dt.dayofyear.astype("int16")
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype("int16")
    out["quarter"] = out["date"].dt.quarter.astype("int8")
    out["is_month_start"] = out["date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")
    days_in_month = out["date"].dt.days_in_month.astype("int16")
    day = out["day"].astype("int16")
    out["is_payday"] = ((day == 15) | (day == days_in_month)).astype("int8")
    out["days_to_next_payday"] = np.where(
        day == 15,
        0,
        np.where(
            day == days_in_month,
            0,
            np.where(day < 15, np.minimum(15 - day, days_in_month - day), days_in_month - day),
        ),
    ).astype("int8")
    out["days_since_prev_payday"] = np.where(
        day == days_in_month,
        0,
        np.where(day >= 15, day - 15, day),
    ).astype("int8")
    out["payday_window_3"] = (
        (out["days_to_next_payday"] <= 3) | (out["days_since_prev_payday"] <= 3)
    ).astype("int8")
    # 用周期编码表达一年中的位置，避免 dayofyear 在年首/年末出现人为断点。
    angle = 2 * np.pi * dayofyear / 365.25
    out["doy_sin"] = np.sin(angle).astype("float32")
    out["doy_cos"] = np.cos(angle).astype("float32")
    return out


def build_target_aggregates(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """基于历史训练段构造可扩展的目标统计特征映射表。"""
    agg_specs = {
        "store_nbr": ["store_nbr"],
        "family": ["family"],
        "store_nbr__family": ["store_nbr", "family"],
        "family__dayofweek": ["family", "dayofweek"],
        "store_nbr__dayofweek": ["store_nbr", "dayofweek"],
        "type": ["type"],
        "city": ["city"],
        "state": ["state"],
        "family__month": ["family", "month"],
        "store_nbr__month": ["store_nbr", "month"],
    }
    mappings: dict[str, pd.DataFrame] = {}
    for name, cols in agg_specs.items():
        grp = train_df.groupby(cols)["sales"].agg(["mean", "std"]).reset_index()
        grp.columns = cols + [f"{name}_sales_mean", f"{name}_sales_std"]
        mappings[name] = grp
    return mappings


def apply_target_aggregates(df: pd.DataFrame, mappings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把预先计算好的目标统计特征按键合并回输入表。"""
    out = df.copy()
    for name, mapping in mappings.items():
        key_cols = name.split("__")
        out = out.merge(mapping, on=key_cols, how="left")
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    group_cols: list[str] | None = None,
    value_col: str = "sales",
    feature_prefix: str | None = None,
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """为目标列或观测列添加 lag / rolling 特征。

    当传入 `history_df` 时，会先把历史段和目标段拼接后再计算，从而让测试期也能
    复用训练期的上下文；滚动统计统一使用 `shift(1)`，避免当前行看到自己的标签值。
    """
    group_cols = group_cols or ["store_nbr", "family"]
    feature_prefix = feature_prefix or value_col
    lags = lags or [1, 7, 14, 28]
    windows = windows or [7, 28]

    target = df.copy()
    target["_is_target"] = 1
    target["_order"] = np.arange(len(target))

    if history_df is None:
        work = target.copy()
    else:
        needed_cols = list(dict.fromkeys(group_cols + ["date", value_col]))
        available_cols = [c for c in needed_cols if c in history_df.columns]
        history = history_df[available_cols].copy()
        history["_is_target"] = 0
        history["_order"] = -1
        work = pd.concat([history, target], ignore_index=True, sort=False)

    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(group_cols + ["date", "_order"]).reset_index(drop=True)

    grouped = work.groupby(group_cols, sort=False)[value_col]
    for lag in lags:
        work[f"{feature_prefix}_lag_{lag}"] = grouped.transform(lambda s, lag=lag: s.shift(lag))

    for window in windows:
        # 先 shift(1) 再 rolling，确保不会把当前时点标签泄露回自己的特征里。
        work[f"{feature_prefix}_roll_mean_{window}"] = grouped.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        work[f"{feature_prefix}_roll_std_{window}"] = grouped.transform(
            lambda s, window=window: s.shift(1).rolling(window=window, min_periods=1).std()
        )

    feat_cols = [
        c
        for c in work.columns
        if c.startswith(f"{feature_prefix}_lag_") or c.startswith(f"{feature_prefix}_roll_")
    ]
    work[feat_cols] = work[feat_cols].fillna(-1)

    if history_df is None:
        return work.sort_values("_order").drop(columns=["_is_target", "_order"])

    return work[work["_is_target"] == 1].sort_values("_order").drop(columns=["_is_target", "_order"])


def build_tft_baseline_features(base_df: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """从主表和辅助表构造 TFT baseline 特征。

    该函数只保留当前 baseline 配置里明确需要的列，保证导出结果可直接进入
    `utils/tft_config.py` 中定义的字段语义体系。
    """
    src_tft_dir = Path(__file__).resolve().parents[1]
    if str(src_tft_dir) not in sys.path:
        sys.path.insert(0, str(src_tft_dir))

    from data.load_data import merge_features

    out = merge_features(
        base=base_df,
        stores=frames["stores"],
        oil=frames["oil"],
        holidays=frames["holidays"],
        transactions=frames["transactions"],
    )
    out = add_date_features(out)

    for col in BASELINE_EXPORT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    return out[BASELINE_EXPORT_COLUMNS].copy()


def main() -> None:
    """导出 train/test baseline 特征 CSV，便于人工检查和后续训练。"""
    src_tft_dir = Path(__file__).resolve().parents[1]
    if str(src_tft_dir) not in sys.path:
        sys.path.insert(0, str(src_tft_dir))

    from data.load_data import load_raw_data

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    output_dir = Path(__file__).resolve().parent

    frames = load_raw_data(data_dir)
    train_features = build_tft_baseline_features(frames["train"], frames)
    test_features = build_tft_baseline_features(frames["test"], frames)

    train_path = output_dir / "train_tft_baseline_features.csv"
    test_path = output_dir / "test_tft_baseline_features.csv"
    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)

    print(f"Wrote {train_path} shape={train_features.shape}")
    print(f"Wrote {test_path} shape={test_features.shape}")


if __name__ == "__main__":
    main()
