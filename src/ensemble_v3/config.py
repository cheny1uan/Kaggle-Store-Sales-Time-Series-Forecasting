from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModeConfig:
    history_days: int | None
    valid_days: int
    sales_estimators: int
    sales_xgb_estimators: int
    tx_estimators: int
    tx_xgb_estimators: int
    early_stopping: int
    full_retrain: bool
    output_name: str
    lgb_seeds: tuple[int, ...]
    xgb_seed: int
    tx_lgb_seeds: tuple[int, ...]
    tx_xgb_seed: int
    use_long_lags: bool = False


MODES: dict[str, ModeConfig] = {
    "fast": ModeConfig(
        history_days=180,
        valid_days=7,
        sales_estimators=1200,
        sales_xgb_estimators=1200,
        tx_estimators=700,
        tx_xgb_estimators=700,
        early_stopping=60,
        full_retrain=False,
        output_name="ensemble_v3_fast.csv",
        lgb_seeds=(42, 2026),
        xgb_seed=43,
        tx_lgb_seeds=(42,),
        tx_xgb_seed=44,
    ),
    "full": ModeConfig(
        history_days=None,
        valid_days=28,
        sales_estimators=4000,
        sales_xgb_estimators=1200,
        tx_estimators=1800,
        tx_xgb_estimators=1200,
        early_stopping=100,
        full_retrain=True,
        output_name="ensemble_v3_full.csv",
        lgb_seeds=(42, 2026),
        xgb_seed=43,
        tx_lgb_seeds=(),
        tx_xgb_seed=44,
    ),
    "recent365": ModeConfig(
        history_days=365,
        valid_days=28,
        sales_estimators=2800,
        sales_xgb_estimators=1200,
        tx_estimators=1200,
        tx_xgb_estimators=1000,
        early_stopping=100,
        full_retrain=True,
        output_name="ensemble_v3_recent365.csv",
        lgb_seeds=(42, 2026),
        xgb_seed=43,
        tx_lgb_seeds=(),
        tx_xgb_seed=44,
    ),
    "v4_fast": ModeConfig(
        history_days=900,
        valid_days=7,
        sales_estimators=1600,
        sales_xgb_estimators=1200,
        tx_estimators=900,
        tx_xgb_estimators=900,
        early_stopping=80,
        full_retrain=False,
        output_name="ensemble_v4_fast.csv",
        lgb_seeds=(42, 2026),
        xgb_seed=43,
        tx_lgb_seeds=(42,),
        tx_xgb_seed=44,
        use_long_lags=True,
    ),
    "v4_full": ModeConfig(
        history_days=None,
        valid_days=28,
        sales_estimators=4200,
        sales_xgb_estimators=1400,
        tx_estimators=1800,
        tx_xgb_estimators=1200,
        early_stopping=120,
        full_retrain=True,
        output_name="ensemble_v4_full.csv",
        lgb_seeds=(42, 2026),
        xgb_seed=43,
        tx_lgb_seeds=(),
        tx_xgb_seed=44,
        use_long_lags=True,
    ),
    "v4_since2015": ModeConfig(
        history_days=960,
        valid_days=28,
        sales_estimators=3600,
        sales_xgb_estimators=1400,
        tx_estimators=1500,
        tx_xgb_estimators=1100,
        early_stopping=120,
        full_retrain=True,
        output_name="ensemble_v4_since2015.csv",
        lgb_seeds=(42, 2026),
        xgb_seed=43,
        tx_lgb_seeds=(),
        tx_xgb_seed=44,
        use_long_lags=True,
    ),
}


@dataclass
class EnsembleV3Config:
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
