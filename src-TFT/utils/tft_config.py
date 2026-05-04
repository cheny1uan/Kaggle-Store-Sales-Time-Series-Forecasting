from __future__ import annotations

"""TFT 字段配置与基础数据预处理。

这个模块集中定义了 Temporal Fusion Transformer 在本项目中使用的：
- 目标列与时间索引列
- 分组键
- 静态特征、已知未来特征、历史观测特征的划分
- baseline CSV 转换为 PyTorch Forecasting 可消费格式的预处理逻辑

它的职责是把“特征表里有哪些列”转换成“模型应如何理解这些列”。
"""

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

# 目标列、相对时间索引列，以及定义一条时间序列身份的分组键。
TARGET = "sales"
TIME_IDX = "time_idx"
GROUP_IDS = ["store_nbr", "family"]

# 静态特征：在同一条序列内部不随时间变化的类别属性。
STATIC_CATEGORICALS = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
STATIC_REALS: list[str] = []

# 已知未来类别特征：进入预测期后仍然可以提前知道的日历类信息。
TIME_VARYING_KNOWN_CATEGORICALS = [
    "month",
    "day",
    "dayofweek",
    "weekofyear",
    "quarter",
]
# 已知未来实数特征：预测期仍可显式提供给模型的数值变量。
TIME_VARYING_KNOWN_REALS = [
    TIME_IDX,
    "onpromotion",
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
TIME_VARYING_UNKNOWN_CATEGORICALS: list[str] = []
# 历史观测实数特征：训练期可以看到，进入未来预测期后无法直接提前知道。
TIME_VARYING_UNKNOWN_REALS = [TARGET, "transactions", "dcoilwtico"]

# 用于进入 TimeSeriesDataSet 前统一做 dtype 规范化的列集合。
CATEGORICAL_COLUMNS = [*GROUP_IDS, "city", "state", "store_type", "cluster", *TIME_VARYING_KNOWN_CATEGORICALS]
REAL_COLUMNS = [
    *STATIC_REALS,
    *TIME_VARYING_KNOWN_REALS,
    *TIME_VARYING_UNKNOWN_REALS,
]
# baseline CSV 期望包含的列顺序，便于导出和后续字段检查保持稳定。
BASELINE_COLUMNS = [
    "id",
    "date",
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
    "onpromotion",
    *TIME_VARYING_KNOWN_CATEGORICALS,
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
    TARGET,
    "transactions",
    "dcoilwtico",
]


@dataclass(frozen=True)
class TFTDatasetConfig:
    """封装构造 `TimeSeriesDataSet` 所需的字段配置。

    这个 dataclass 统一管理目标列、分组键、特征分类和窗口长度，避免训练脚本、
    推理脚本和 dataset 构造逻辑各自维护一套配置，导致字段语义不一致。
    """

    target: str = TARGET
    time_idx: str = TIME_IDX
    group_ids: tuple[str, ...] = tuple(GROUP_IDS)
    static_categoricals: tuple[str, ...] = tuple(STATIC_CATEGORICALS)
    static_reals: tuple[str, ...] = tuple(STATIC_REALS)
    time_varying_known_categoricals: tuple[str, ...] = tuple(TIME_VARYING_KNOWN_CATEGORICALS)
    time_varying_known_reals: tuple[str, ...] = tuple(TIME_VARYING_KNOWN_REALS)
    time_varying_unknown_categoricals: tuple[str, ...] = tuple(TIME_VARYING_UNKNOWN_CATEGORICALS)
    time_varying_unknown_reals: tuple[str, ...] = tuple(TIME_VARYING_UNKNOWN_REALS)
    max_encoder_length: int = 90
    max_prediction_length: int = 16
    allow_missing_timesteps: bool = True

    def to_dataset_kwargs(self) -> dict[str, Any]:
        """转换为 `TimeSeriesDataSet` 可直接展开的关键字参数。"""
        return {
            "time_idx": self.time_idx,
            "target": self.target,
            "group_ids": list(self.group_ids),
            "static_categoricals": list(self.static_categoricals),
            "static_reals": list(self.static_reals),
            "time_varying_known_categoricals": list(self.time_varying_known_categoricals),
            "time_varying_known_reals": list(self.time_varying_known_reals),
            "time_varying_unknown_categoricals": list(self.time_varying_unknown_categoricals),
            "time_varying_unknown_reals": list(self.time_varying_unknown_reals),
            "allow_missing_timesteps": self.allow_missing_timesteps,
        }

    def to_dict(self) -> dict[str, Any]:
        """将配置序列化为普通字典，便于日志或调试输出。"""
        return asdict(self)


def add_time_idx(df: pd.DataFrame, reference_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """为输入表添加相对天数形式的时间索引。

    TFT 更适合消费连续整数时间索引，而不是直接使用 datetime。这里以给定参考日
    或数据中的最早日期为 0 点，将每一行映射为距离参考日的天数。
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    start = pd.Timestamp(reference_date) if reference_date is not None else out["date"].min()
    out[TIME_IDX] = (out["date"] - start).dt.days.astype("int32")
    return out


def prepare_tft_dataframe(df: pd.DataFrame, reference_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    """把 baseline DataFrame 规范化为 TFT 可直接使用的格式。

    这里会统一补充 `time_idx`，并把类别列 / 数值列强制转换为稳定 dtype。
    其中 `type` 会被重命名为 `store_type`，以避免和 PyTorch 模块已有属性名冲突。
    """
    out = add_time_idx(df, reference_date=reference_date)
    if "type" in out.columns and "store_type" not in out.columns:
        out = out.rename(columns={"type": "store_type"})

    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype("string")

    for col in REAL_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def get_default_tft_config() -> TFTDatasetConfig:
    """返回项目默认的 TFT dataset 配置。"""
    return TFTDatasetConfig()
