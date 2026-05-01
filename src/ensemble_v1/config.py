from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModeConfig:
    history_days: int | None
    valid_days: int
    lgb_estimators: int
    xgb_estimators: int
    early_stopping: int
    full_retrain: bool
    output_name: str


MODES: dict[str, ModeConfig] = {
    "fast": ModeConfig(
        history_days=180,
        valid_days=7,
        lgb_estimators=800,
        xgb_estimators=500,
        early_stopping=50,
        full_retrain=False,
        output_name="ensemble_v1_fast.csv",
    ),
    "full": ModeConfig(
        history_days=None,
        valid_days=28,
        lgb_estimators=4000,
        xgb_estimators=1200,
        early_stopping=100,
        full_retrain=True,
        output_name="ensemble_v1_full.csv",
    ),
}


@dataclass
class EnsembleV1Config:
    data_dir: str = "data"
    output_dir: str = "outputs"
    submission_dir: str = "submissions"
    mode: str = "fast"
    feature_drop: set[str] = field(
        default_factory=lambda: {
            "bridge_flag",
            "workday_flag",
            "year",
            "quarter",
            "holiday_any",
            "holiday_flag",
            "is_month_end",
            "is_month_start",
            "additional_flag",
            "holiday_regional",
            "holiday_cnt",
        }
    )

