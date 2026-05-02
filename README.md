# Store Sales Time Series Forecasting

Machine learning course project for Kaggle Store Sales - Time Series Forecasting.

## Current Versions

- `ensemble_v1`: verified final baseline, public score **0.42660**
- `ensemble_v2`: current optimization candidate, built on recursive forecasting and multi-seed blending

## Current Best Local Result

- Final local submission file: `submissions/ensemble_v2_full.csv`
- Recursive validation RMSLE: **0.40638**

## Method

The solution is a tabular time-series ensemble:

- LightGBM + XGBoost regression on `log1p(sales)`
- Day-by-day recursive forecasting for the 15-day test horizon
- Sales lag and rolling-window features
- Zero forecasting for store-family pairs that never sold historically
- Oil price interpolation and oil-related features
- Holiday, payday, earthquake, date, and store metadata features
- Static sales aggregate features by store, family, city, state, type, and month
- `transactions` features are excluded from final model inputs because future test transactions are unavailable

## Project Structure

```text
.
├── data/                         # Kaggle raw CSV files, ignored by Git
├── notebooks/                    # Optional exploration notebooks
├── outputs/                      # Optional generated outputs, ignored by Git
├── scripts/
│   └── run_ensemble_v2.py        # Current training and submission entry point
├── src/
│   ├── data/
│   │   └── load_data.py          # Read and merge raw Kaggle tables
│   ├── ensemble_v1/
│   │   ├── config.py             # Verified baseline settings
│   │   ├── ensemble.py           # Prediction blending utilities
│   │   ├── features.py           # Dataset assembly and zero forecasting
│   │   ├── models.py             # LightGBM/XGBoost training and prediction
│   │   └── pipeline.py           # Verified baseline pipeline
│   ├── ensemble_v2/
│   │   ├── config.py             # V2 multi-seed settings
│   │   ├── ensemble.py           # V2 prediction blending utilities
│   │   ├── features.py           # V2 promo and holiday features
│   │   ├── models.py             # V2 multi-seed model training
│   │   └── pipeline.py           # V2 recursive ensemble pipeline
│   ├── features/
│   │   └── make_features.py      # Shared date, aggregate, lag/rolling features
│   ├── models/
│   │   └── train_lgbm.py         # Shared LightGBM helpers
│   └── utils/
│       └── metrics.py            # RMSLE metric
├── submissions/
│   └── ensemble_v2_full.csv      # Current local submission file, ignored by Git
├── README.md
└── requirements.txt
```

## How To Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Place Kaggle data files in `data/`:

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
```

Fast debug run:

```powershell
python scripts/run_ensemble_v2.py --mode fast
python scripts/run_ensemble_v2.py --mode full
```

The final submission is saved to:

```text
submissions/ensemble_v2_full.csv
```

## Notes

Raw data, generated outputs, and submissions are ignored by Git to avoid large-file upload problems. The final CSV is kept locally for Kaggle submission.
