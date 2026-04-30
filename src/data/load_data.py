from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw_data(data_dir: str | Path) -> dict[str, pd.DataFrame]:
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
    df = holidays.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["transferred"] = df["transferred"].astype(int)
    df["is_holiday"] = (df["type"] == "Holiday").astype(int)
    df["is_event"] = (df["type"] == "Event").astype(int)
    df["is_additional"] = (df["type"] == "Additional").astype(int)
    df["is_transfer"] = (df["type"] == "Transfer").astype(int)
    df["is_bridge"] = (df["type"] == "Bridge").astype(int)
    df["is_work_day"] = (df["type"] == "Work Day").astype(int)
    grouped = df.groupby("date", as_index=False).agg(
        holiday_cnt=("date", "size"),
        holiday_any=("date", "size"),
        holiday_national=("locale", lambda s: (s == "National").sum()),
        holiday_regional=("locale", lambda s: (s == "Regional").sum()),
        holiday_local=("locale", lambda s: (s == "Local").sum()),
        holiday_transferred=("transferred", "sum"),
        holiday_flag=("is_holiday", "max"),
        event_flag=("is_event", "max"),
        additional_flag=("is_additional", "max"),
        transfer_flag=("is_transfer", "max"),
        bridge_flag=("is_bridge", "max"),
        workday_flag=("is_work_day", "max"),
    )
    return grouped


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby(["date", "store_nbr"], as_index=False)["transactions"].sum()


def build_oil_features(oil: pd.DataFrame) -> pd.DataFrame:
    df = oil.copy()
    df["date"] = pd.to_datetime(df["date"])
    full_dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), df["date"].max(), freq="D")})
    df = full_dates.merge(df, on="date", how="left")
    df["dcoilwtico_raw"] = df["dcoilwtico"]
    df["oil_missing"] = df["dcoilwtico"].isna().astype("int8")
    df["dcoilwtico"] = df["dcoilwtico"].interpolate(method="linear", limit_direction="both")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    df["oil_diff_1"] = df["dcoilwtico"].diff().fillna(0)
    df["oil_pct_change_1"] = df["dcoilwtico"].pct_change().replace([float("inf"), float("-inf")], 0).fillna(0)
    df["expensive_oil"] = (df["dcoilwtico"] >= 60).astype("int8")
    return df


def merge_features(
    base: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
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
    out["transactions"] = out["transactions"].fillna(0)
    return out
