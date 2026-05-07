from __future__ import annotations

"""TFT 模型工厂。

这个模块把 PyTorch Forecasting 的 `TemporalFusionTransformer` 封装成项目内统一的
创建与 checkpoint 加载入口，避免训练脚本和推理脚本重复维护模型构造细节。
"""

from pathlib import Path

from pytorch_forecasting import TemporalFusionTransformer

from utils.metrics import RMSLELoss


def build_tft_model(
    training_dataset,
    learning_rate: float = 0.03,
    hidden_size: int = 32,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    hidden_continuous_size: int = 16,
    lstm_layers: int = 1,
    reduce_on_plateau_patience: int = 3,
) -> TemporalFusionTransformer:
    """基于训练 dataset 的字段定义构造 TFT 模型。

    当前配置使用单点预测头，每个未来步只输出一个 sales 预测值，
    并直接以 RMSLE 作为训练目标。
    """
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        lstm_layers=lstm_layers,
        output_size=1,
        loss=RMSLELoss(),
        reduce_on_plateau_patience=reduce_on_plateau_patience,
    )


def load_tft_from_checkpoint(
    checkpoint_path: str | Path,
    learning_rate: float | None = None,
    reduce_on_plateau_patience: int | None = None,
) -> TemporalFusionTransformer:
    """从 checkpoint 恢复训练好的 TFT 模型。

    这里支持覆盖学习率等训练超参数，便于把已有 best checkpoint 作为权重初始化，
    再在新的训练集或新的训练轮次上继续训练。
    """
    load_kwargs = {}
    if learning_rate is not None:
        load_kwargs["learning_rate"] = learning_rate
    if reduce_on_plateau_patience is not None:
        load_kwargs["reduce_on_plateau_patience"] = reduce_on_plateau_patience
    return TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_path), **load_kwargs)
