from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype("int16")
    out["month"] = out["date"].dt.month.astype("int8")
    out["day"] = out["date"].dt.day.astype("int8")
    out["dayofweek"] = out["date"].dt.dayofweek.astype("int8")
    out["dayofyear"] = out["date"].dt.dayofyear.astype("int16")
    out["weekofyear"] = out["date"].dt.isocalendar().week.astype("int16")
    out["quarter"] = out["date"].dt.quarter.astype("int8")
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")
    out["is_month_start"] = out["date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")

    days_in_month = out["date"].dt.days_in_month.astype("int16")
    day = out["day"].astype("int16")
    out["is_payday"] = ((day == 15) | (day == days_in_month)).astype("int8")
    out["days_to_next_payday"] = np.where(day < 15, 15 - day, days_in_month - day).astype("int8")
    out["days_since_prev_payday"] = np.where(day >= 15, day - 15, day).astype("int8")
    out["payday_window_3"] = (
        (out["days_to_next_payday"] <= 3) | (out["days_since_prev_payday"] <= 3)
    ).astype("int8")

    earthquake = pd.Timestamp("2016-04-16")
    delta = (out["date"] - earthquake).dt.days
    out["is_earthquake_day"] = (delta == 0).astype("int8")
    out["earthquake_window_7"] = delta.between(0, 7).astype("int8")
    out["earthquake_window_30"] = delta.between(0, 30).astype("int8")

    angle = 2 * np.pi * out["dayofyear"] / 365.25
    out["doy_sin"] = np.sin(angle).astype("float32")
    out["doy_cos"] = np.cos(angle).astype("float32")
    return out


def add_fourier_terms(df: pd.DataFrame, period: int, order: int, prefix: str) -> pd.DataFrame:
    out = df.copy()
    t = (pd.to_datetime(out["date"]) - pd.Timestamp("2013-01-01")).dt.days.astype(float)
    for k in range(1, order + 1):
        out[f"{prefix}_sin_{k}"] = np.sin(2 * np.pi * k * t / period).astype("float32")
        out[f"{prefix}_cos_{k}"] = np.cos(2 * np.pi * k * t / period).astype("float32")
    return out


def merge_external_features(base: pd.DataFrame, stores: pd.DataFrame, oil: pd.DataFrame, holidays: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["date"] = pd.to_datetime(out["date"])
    stores = stores.copy()

    oil = oil.copy()
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.sort_values("date")
    full_dates = pd.DataFrame({"date": pd.date_range(oil["date"].min(), oil["date"].max(), freq="D")})
    oil = full_dates.merge(oil, on="date", how="left")
    oil["dcoilwtico_raw"] = oil["dcoilwtico"]
    oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(limit_direction="both").ffill().bfill()
    oil["oil_diff_1"] = oil["dcoilwtico"].diff().fillna(0)
    oil["oil_pct_change_1"] = oil["dcoilwtico"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    oil["expensive_oil"] = (oil["dcoilwtico"] >= 60).astype("int8")

    holidays = holidays.copy()
    holidays["date"] = pd.to_datetime(holidays["date"])
    holidays["transferred"] = holidays["transferred"].astype(int)
    holiday_daily = holidays.groupby("date", as_index=False).agg(
        holiday_national=("locale", lambda s: (s == "National").sum()),
        holiday_regional=("locale", lambda s: (s == "Regional").sum()),
        holiday_local=("locale", lambda s: (s == "Local").sum()),
        holiday_flag=("type", lambda s: (s == "Holiday").max()),
        event_flag=("type", lambda s: (s == "Event").max()),
        transfer_flag=("type", lambda s: (s == "Transfer").max()),
        bridge_flag=("type", lambda s: (s == "Bridge").max()),
        workday_flag=("type", lambda s: (s == "Work Day").max()),
        holiday_cnt=("date", "size"),
        holiday_any=("date", "size"),
        additional_flag=("type", lambda s: (s == "Additional").max()),
    )

    transactions = transactions.copy()
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions = transactions.groupby(["date", "store_nbr"], as_index=False)["transactions"].sum()

    out = out.merge(stores, on="store_nbr", how="left")
    out = out.merge(oil, on="date", how="left")
    out = out.merge(holiday_daily, on="date", how="left")
    out = out.merge(transactions, on=["date", "store_nbr"], how="left")
    out["transactions"] = out["transactions"].fillna(0)
    return out

