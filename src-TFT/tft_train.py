from __future__ import annotations

"""TFT 训练入口脚本。

这个脚本负责把 baseline 特征 CSV 接入 PyTorch Forecasting 的
Temporal Fusion Transformer 训练流程，完成：
- 输入 train/test 特征文件的合理性校验
- dataset / dataloader 构建
- 模型实例化
- Lightning Trainer 配置
- checkpoint 保存
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

SRC_TFT_DIR = Path(__file__).resolve().parent
if str(SRC_TFT_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_TFT_DIR))

from data.tft_dataloaders import build_tft_dataloaders, load_baseline_feature_frames
from models.tft_model import build_tft_model
from utils.tft_config import TFTDatasetConfig


def parse_args() -> argparse.Namespace:
    """解析训练脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a TFT model with PyTorch Forecasting.")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to historical train baseline CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to future test baseline CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-length", type=int, default=90)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--attention-head-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-continuous-size", type=int, default=16)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="1")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--gradient-clip-val", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=str(SRC_TFT_DIR / "checkpoints"))
    return parser.parse_args()


def validate_feature_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """校验输入文件是否符合“历史训练段 + 未来测试段”的语义。

    这里的目标不是修复数据，而是在脚本入口尽早阻止把 train/test 特征文件传反，
    否则后续训练与推理虽然可能继续执行，但语义已经完全错误。
    """
    train_has_sales = train_df["sales"].notna().mean() > 0.95
    test_has_missing_sales = test_df["sales"].isna().mean() > 0.95
    train_max_date = pd.to_datetime(train_df["date"]).max()
    test_min_date = pd.to_datetime(test_df["date"]).min()

    if train_has_sales and test_has_missing_sales and train_max_date <= test_min_date:
        return

    raise ValueError(
        "The provided train/test feature CSVs do not look like historical-train then future-test data. "
        "Current exported defaults appear swapped. Pass the historical file via --train-csv and the future file via --test-csv."
    )


def build_trainer(args: argparse.Namespace) -> tuple[Trainer, ModelCheckpoint]:
    """根据命令行参数构造 Lightning Trainer 与 checkpoint 回调。"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    callbacks = [
        checkpoint_callback,
        # 以验证集损失为准提前停止，避免无效继续训练。
        EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, mode="min"),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        enable_checkpointing=True,
        enable_progress_bar=True,
    )
    return trainer, checkpoint_callback


def main() -> None:
    """串联数据校验、dataloader 构建、模型训练与 checkpoint 输出。"""
    args = parse_args()
    seed_everything(42, workers=True)

    train_df, test_df = load_baseline_feature_frames(args.train_csv, args.test_csv)
    validate_feature_split(train_df, test_df)

    cfg = TFTDatasetConfig(
        max_encoder_length=args.encoder_length,
        max_prediction_length=args.prediction_length,
    )
    dataloaders = build_tft_dataloaders(
        train_df=train_df,
        test_df=test_df,
        config=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = build_tft_model(
        dataloaders.training,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        hidden_continuous_size=args.hidden_continuous_size,
        lstm_layers=args.lstm_layers,
        reduce_on_plateau_patience=max(1, args.early_stopping_patience // 2),
    )
    trainer, checkpoint_callback = build_trainer(args)
    trainer.fit(
        model,
        train_dataloaders=dataloaders.train_dataloader,
        val_dataloaders=dataloaders.val_dataloader,
    )

    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
