from __future__ import annotations

"""构建 TFT 训练、验证和预测用的 dataset / dataloader。

这个模块负责把 baseline 特征 CSV 转成 PyTorch Forecasting 的
`TimeSeriesDataSet`，并进一步封装为训练、验证和预测 dataloader。

这里最关键的逻辑包括：
- 历史训练段与未来测试段的统一预处理
- 训练/验证/预测三类数据集的构造差异
- 为预测窗口拼接每个序列最近的编码器历史，保证推理时上下文完整
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import DataLoader

SRC_TFT_DIR = Path(__file__).resolve().parents[1]
if str(SRC_TFT_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_TFT_DIR))

from utils.tft_config import TARGET, TIME_IDX, TFTDatasetConfig, get_default_tft_config, prepare_tft_dataframe

try:
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
except ImportError as exc:  # pragma: no cover - import guard for local setup differences
    TimeSeriesDataSet = None
    GroupNormalizer = None
    PYTORCH_FORECASTING_IMPORT_ERROR = exc
else:
    PYTORCH_FORECASTING_IMPORT_ERROR = None


@dataclass
class TFTDataLoaders:
    """打包训练、验证、预测三套 dataset 与 dataloader。"""

    training: Any
    validation: Any
    prediction: Any
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    predict_dataloader: DataLoader


def _require_pytorch_forecasting() -> None:
    """在真正构造 dataset 前显式检查依赖是否可用。"""
    if PYTORCH_FORECASTING_IMPORT_ERROR is not None:
        raise ImportError(
            "pytorch-forecasting is required to build TFT datasets/dataloaders. "
            "Install pytorch-forecasting and lightning in the active environment first."
        ) from PYTORCH_FORECASTING_IMPORT_ERROR


def default_feature_paths() -> tuple[Path, Path]:
    """返回默认的 baseline train/test 特征文件路径。"""
    feature_dir = SRC_TFT_DIR / "features"
    return feature_dir / "train_tft_baseline_features.csv", feature_dir / "test_tft_baseline_features.csv"


def load_baseline_feature_frames(
    train_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 baseline train/test 特征表。

    若未显式传入路径，则默认读取 `src-TFT/features` 下的导出结果。
    """
    default_train_csv, default_test_csv = default_feature_paths()
    train_path = Path(train_csv) if train_csv is not None else default_train_csv
    test_path = Path(test_csv) if test_csv is not None else default_test_csv

    train_df = pd.read_csv(train_path, parse_dates=["date"])
    test_df = pd.read_csv(test_path, parse_dates=["date"])
    return train_df, test_df


def prepare_train_val_test_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reference_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """统一整理训练段、测试段和预测拼接表。

    训练段保留真实标签；测试段会填入推理阶段所需的占位值，保证
    `TimeSeriesDataSet` 能够构造完整窗口；返回的第三个 DataFrame 仅用于
    需要同时观察历史与未来的预测场景。
    """
    ref_date = pd.Timestamp(reference_date) if reference_date is not None else pd.to_datetime(train_df["date"]).min()

    train_prepared = prepare_tft_dataframe(train_df, reference_date=ref_date)
    test_prepared = prepare_tft_dataframe(test_df, reference_date=ref_date)

    # 预测期没有真实销量和未来交易量，这里使用占位值让窗口构造继续成立。
    if TARGET in test_prepared.columns:
        test_prepared[TARGET] = 0.0
    if "transactions" in test_prepared.columns:
        test_prepared["transactions"] = 0.0
    if "dcoilwtico" in test_prepared.columns:
        # 油价按“最后一个已知有效值延续”的方案填充未来缺失值。
        last_oil = train_prepared.sort_values("date")["dcoilwtico"].dropna().iloc[-1]
        test_prepared["dcoilwtico"] = test_prepared["dcoilwtico"].fillna(last_oil)

    predict_frame = pd.concat([train_prepared, test_prepared], ignore_index=True, sort=False)
    predict_frame = predict_frame.sort_values([*get_default_tft_config().group_ids, "date"]).reset_index(drop=True)
    return train_prepared, test_prepared, predict_frame


def build_tft_datasets(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TFTDatasetConfig | None = None,
):
    """构造训练、验证、预测三套 `TimeSeriesDataSet`。

    训练集只使用历史段中能完整形成编码器窗口和预测窗口的样本；验证集复用训练集
    的字段定义，但在历史段末尾按 `predict=True` 方式抽取预测窗口；预测集则需要
    额外拼接每个序列最近的历史尾部，作为未来测试段的编码器上下文。
    """
    _require_pytorch_forecasting()
    cfg = config or get_default_tft_config()

    train_prepared, test_prepared, _ = prepare_train_val_test_frames(train_df, test_df)
    # 留出最后一个预测窗口长度作为验证/预测区域，避免训练时直接看到最末端标签。
    training_cutoff = int(train_prepared[TIME_IDX].max()) - cfg.max_prediction_length

    training = TimeSeriesDataSet(
        train_prepared[train_prepared[TIME_IDX] <= training_cutoff].copy(),
        min_encoder_length=cfg.max_encoder_length,
        max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=cfg.max_prediction_length,
        max_prediction_length=cfg.max_prediction_length,
        target_normalizer=GroupNormalizer(groups=list(cfg.group_ids), transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        **cfg.to_dataset_kwargs(),
    )
    validation = TimeSeriesDataSet.from_dataset(
        training,
        train_prepared,
        predict=True,
        stop_randomization=True,
    )

    # 预测未来测试段时，每条序列都必须带上最近一个 encoder 窗口长度的历史上下文。
    history_tail = train_prepared.groupby(list(cfg.group_ids), group_keys=False).tail(cfg.max_encoder_length)
    predict_frame = pd.concat([history_tail, test_prepared], ignore_index=True, sort=False)
    predict_frame = predict_frame.sort_values([*cfg.group_ids, "date"]).reset_index(drop=True)
    predict_frame[TARGET] = predict_frame[TARGET].astype("float32")
    prediction = TimeSeriesDataSet.from_dataset(
        training,
        predict_frame,
        predict=True,
        stop_randomization=True,
    )
    return training, validation, prediction


def build_tft_dataloaders(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: TFTDatasetConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> TFTDataLoaders:
    """基于三套 dataset 构造对应 dataloader。"""
    training, validation, prediction = build_tft_datasets(train_df, test_df, config=config)
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    # 验证和推理不做反向传播，通常可以使用更大的 batch size 提高吞吐。
    eval_batch_size = max(batch_size, 1) * 4
    val_dataloader = validation.to_dataloader(train=False, batch_size=eval_batch_size, num_workers=num_workers)
    predict_dataloader = prediction.to_dataloader(train=False, batch_size=eval_batch_size, num_workers=num_workers)
    return TFTDataLoaders(
        training=training,
        validation=validation,
        prediction=prediction,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        predict_dataloader=predict_dataloader,
    )


def build_from_csv(
    train_csv: str | Path | None = None,
    test_csv: str | Path | None = None,
    config: TFTDatasetConfig | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> TFTDataLoaders:
    """从 CSV 文件直接构造 TFT dataloader，方便命令行和脚本复用。"""
    train_df, test_df = load_baseline_feature_frames(train_csv=train_csv, test_csv=test_csv)
    return build_tft_dataloaders(
        train_df=train_df,
        test_df=test_df,
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def parse_args() -> argparse.Namespace:
    """解析 dataloader smoke test 命令行参数。"""
    parser = argparse.ArgumentParser(description="Build PyTorch Forecasting TFT datasets and dataloaders.")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to train_tft_baseline_features.csv")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test_tft_baseline_features.csv")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for dataloaders")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--encoder-length", type=int, default=90, help="Max encoder length")
    parser.add_argument("--prediction-length", type=int, default=16, help="Max prediction length")
    return parser.parse_args()


def main() -> None:
    """打印 dataset / dataloader 的窗口与 batch 数量，用于轻量验证。"""
    args = parse_args()
    cfg = TFTDatasetConfig(
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
    )
    dataloaders = build_from_csv(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        config=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"training windows: {len(dataloaders.training)}")
    print(f"validation windows: {len(dataloaders.validation)}")
    print(f"prediction windows: {len(dataloaders.prediction)}")
    print(f"train batches: {len(dataloaders.train_dataloader)}")
    print(f"validation batches: {len(dataloaders.val_dataloader)}")
    print(f"prediction batches: {len(dataloaders.predict_dataloader)}")


if __name__ == "__main__":
    main()
