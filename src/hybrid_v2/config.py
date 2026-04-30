from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HybridV2Config:
    data_dir: str = "data"
    output_dir: str = "outputs"
    submission_dir: str = "submissions"
    valid_days: int = 28
    fast_history_days: int | None = 180
    fast_n_estimators: int = 800
    full_n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 128
    min_child_samples: int = 50
    early_stopping_rounds: int = 100
    trend_fourier_orders: tuple[int, ...] = (1, 3, 5)
    trend_alpha: float = 50.0
    lags: tuple[int, ...] = (1, 7, 14, 28, 35, 42, 56)
    rolling_windows: tuple[int, ...] = (7, 28)
    zero_guard_onpromotion: bool = True
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
