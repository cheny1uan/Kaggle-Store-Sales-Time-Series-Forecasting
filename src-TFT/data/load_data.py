from __future__ import annotations

"""读取原始 CSV 并构造 baseline merge 所需的基础辅助特征。

这里的职责比 `features/make_features.py` 更靠近数据源一层：
- 读取 Kaggle 原始表
- 对节假日、交易量、油价做基础聚合或补齐
- 将辅助表按键拼接回主表

该模块不负责定义 TFT 的字段分组，只负责产出干净的基础输入。
"""

from pathlib import Path

import pandas as pd


def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """读取项目原始 CSV，并对包含 `date` 的表自动做日期解析。"""
    data_dir = Path(data_dir)
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
        "sample_submission": "sample_submission.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    for key, filename in files.items():
        path = data_dir / filename
        parse_dates = ["date"] if "date" in pd.read_csv(path, nrows=0).columns else None
        frames[key] = pd.read_csv(path, parse_dates=parse_dates)
    return frames


def build_holiday_features(holidays: pd.DataFrame) -> pd.DataFrame:
    """把原始节假日表压缩为按日期聚合的 holiday/event 标记。"""
    df = holidays.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["is_holiday"] = ((df["type"] == "Holiday") & (~df["transferred"].astype(bool))).astype("int8")
    df["is_event"] = (df["type"] == "Event").astype("int8")
    grouped = df.groupby("date", as_index=False).agg(
        holiday_flag=("is_holiday", "max"),
        event_flag=("is_event", "max"),
    )
    return grouped


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """按日期和门店聚合交易量，作为门店级历史观测特征。"""
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby(["date", "store_nbr"], as_index=False)["transactions"].sum()


def build_oil_features(oil: pd.DataFrame) -> pd.DataFrame:
    """把油价序列补齐到日频，并用插值与前后填充消除缺口。"""
    df = oil.copy()
    df["date"] = pd.to_datetime(df["date"])
    full_dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    df = full_dates.merge(df, on="date", how="left")
    df["dcoilwtico"] = df["dcoilwtico"].interpolate(method="linear", limit_direction="both")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    return df[["date", "dcoilwtico"]]


def merge_features(
    base: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """把门店、油价、节假日、交易量等辅助信息拼接回主表。"""
    out = base.copy()
    out["date"] = pd.to_datetime(out["date"])
    stores = stores.copy()
    oil = build_oil_features(oil)
    holidays = build_holiday_features(holidays)
    transactions = build_transaction_features(transactions)
    out = out.merge(stores, on="store_nbr", how="left")
    out = out.merge(oil, on="date", how="left")
    out = out.merge(holidays, on="date", how="left")
    out = out.merge(transactions, on=["date", "store_nbr"], how="left")
    # 没有命中节假日/事件/交易量记录时，按“无事件、无记录即 0”处理。
    out["holiday_flag"] = out["holiday_flag"].fillna(0).astype("int8")
    out["event_flag"] = out["event_flag"].fillna(0).astype("int8")
    out["transactions"] = out["transactions"].fillna(0)
    return out



def main() -> None:
    """导出 merge 后的 train/test CSV，便于检查原始辅助特征拼接结果。"""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    output_dir = Path(__file__).resolve().parent

    frames = load_raw_data(data_dir)
    common_kwargs = {
        "stores": frames["stores"],
        "oil": frames["oil"],
        "holidays": frames["holidays"],
        "transactions": frames["transactions"],
    }

    train_merged = merge_features(base=frames["train"], **common_kwargs)
    test_merged = merge_features(base=frames["test"], **common_kwargs)

    train_path = output_dir / "train_merged.csv"
    test_path = output_dir / "test_merged.csv"
    train_merged.to_csv(train_path, index=False)
    test_merged.to_csv(test_path, index=False)

    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")


if __name__ == "__main__":
    main()
