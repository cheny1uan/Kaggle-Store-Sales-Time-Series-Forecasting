from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_dir = Path(data_dir)
    mapping = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
        "sample_submission": "sample_submission.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in mapping.items():
        path = data_dir / filename
        parse_dates = ["date"] if "date" in pd.read_csv(path, nrows=0).columns else None
        tables[key] = pd.read_csv(path, parse_dates=parse_dates)
    return tables


def save_submission(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

