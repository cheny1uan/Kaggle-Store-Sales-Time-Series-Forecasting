from __future__ import annotations

"""TFT 推理入口：从 checkpoint 生成 Kaggle submission。

脚本流程为：
1. 读取历史训练特征与未来测试特征
2. 复用训练时的 dataset 配置构建 prediction dataloader
3. 加载 TFT checkpoint 并输出多步预测结果
4. 将预测窗口展开为逐行结果
5. 按测试集语义键和 sample submission 的 id 顺序回贴预测值
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from lightning.pytorch import seed_everything

SRC_TFT_DIR = Path(__file__).resolve().parent
if str(SRC_TFT_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_TFT_DIR))

from data.tft_dataloaders import build_tft_dataloaders, load_baseline_feature_frames
from models.tft_model import load_tft_from_checkpoint
from tft_train import validate_feature_split
from utils.tft_config import TARGET, TIME_IDX, TFTDatasetConfig, prepare_tft_dataframe


def parse_args() -> argparse.Namespace:
    """解析推理脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="Run TFT inference and write a Kaggle submission CSV.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained TFT checkpoint")
    parser.add_argument("--train-csv", type=str, default=None, help="Path to historical train baseline CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to future test baseline CSV")
    parser.add_argument(
        "--sample-submission",
        type=str,
        default=str(SRC_TFT_DIR.parent / "data" / "sample_submission.csv"),
        help="Path to Kaggle sample_submission.csv",
    )
    parser.add_argument("--output", type=str, default=str(SRC_TFT_DIR / "submission_tft.csv"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--encoder-length", type=int, default=90)
    parser.add_argument("--prediction-length", type=int, default=16)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="1")
    return parser.parse_args()


def flatten_predictions(raw_predictions) -> pd.DataFrame:
    """把 TFT 多步预测结果展开成逐行表格。

    PyTorch Forecasting 的预测结果按“样本 × 预测步”组织，这里会把它改写为
    `store_nbr/family/time_idx/sales` 形式，便于后续按语义键回贴到测试集行上。
    """
    prediction_obj = raw_predictions.output if hasattr(raw_predictions, "output") else raw_predictions
    if hasattr(prediction_obj, "detach"):
        prediction_array = prediction_obj.detach().cpu().numpy()
    else:
        prediction_array = prediction_obj.cpu().numpy()

    if prediction_array.ndim == 3 and prediction_array.shape[-1] == 1:
        prediction_array = prediction_array[..., 0]
    elif prediction_array.ndim == 3:
        prediction_array = prediction_array.mean(axis=-1)

    if prediction_array.ndim != 2:
        raise ValueError(f"Unexpected prediction shape: {prediction_array.shape}")

    index_df = raw_predictions.index.copy()
    decoder_lengths = raw_predictions.decoder_lengths.cpu().numpy()

    rows: list[dict[str, object]] = []
    for i, row in index_df.iterrows():
        # 这里的 time_idx 已经对应 decoder 起点，逐步展开即可映射到测试集每日行。
        decoder_start = int(row[TIME_IDX])
        decoder_length = int(decoder_lengths[i])
        for step in range(decoder_length):
            rows.append(
                {
                    "store_nbr": str(row["store_nbr"]),
                    "family": str(row["family"]),
                    TIME_IDX: decoder_start + step,
                    TARGET: max(float(prediction_array[i, step]), 0.0),
                }
            )
    return pd.DataFrame(rows)


def build_submission(pred_rows: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame, sample_submission_path: str | Path) -> pd.DataFrame:
    """把展开后的预测结果对齐回 Kaggle submission 格式。"""
    reference_date = pd.to_datetime(train_df["date"]).min()
    test_prepared = prepare_tft_dataframe(test_df, reference_date=reference_date)
    test_prepared["store_nbr"] = test_prepared["store_nbr"].astype("string")
    test_prepared["family"] = test_prepared["family"].astype("string")

    submission_base = test_prepared[["id", "store_nbr", "family", TIME_IDX, "date"]].copy()
    # 先按语义键对齐预测值，再按 sample submission 的 id 顺序输出最终结果。
    merged = submission_base.merge(
        pred_rows,
        on=["store_nbr", "family", TIME_IDX],
        how="left",
        validate="one_to_one",
    )
    if merged[TARGET].isna().any():
        raise ValueError("Missing predicted sales after aligning predictions back to the test rows.")

    sample_submission = pd.read_csv(sample_submission_path)
    submission = sample_submission[["id"]].merge(
        merged[["id", TARGET]],
        on="id",
        how="left",
        validate="one_to_one",
    )
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count does not match sample submission.")
    if submission[TARGET].isna().any():
        raise ValueError("Submission contains missing sales values.")

    submission[TARGET] = submission[TARGET].clip(lower=0)
    return submission


def main() -> None:
    """执行 TFT 推理并输出 submission CSV。"""
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
    model = load_tft_from_checkpoint(args.checkpoint)
    raw_predictions = model.predict(
        dataloaders.predict_dataloader,
        mode="prediction",
        return_index=True,
        return_decoder_lengths=True,
        trainer_kwargs={"accelerator": args.accelerator, "devices": args.devices, "enable_progress_bar": True},
    )
    pred_rows = flatten_predictions(raw_predictions)
    submission = build_submission(pred_rows, train_df, test_df, args.sample_submission)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
