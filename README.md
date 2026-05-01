# Store Sales Time Series Forecasting

Final version for the machine learning course project.

## Final Result

- Competition: Kaggle Store Sales - Time Series Forecasting
- Best public score: **0.42660**
- Final submission file: `submissions/ensemble_v1_full.csv`
- Final model version: `ensemble_v1`

## Method

The final solution is a tabular time-series ensemble:

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
│   └── run_ensemble_v1.py        # Final training and submission entry point
├── src/
│   ├── data/
│   │   └── load_data.py          # Read and merge raw Kaggle tables
│   ├── ensemble_v1/
│   │   ├── config.py             # Fast/full mode settings
│   │   ├── ensemble.py           # Prediction blending utilities
│   │   ├── features.py           # Dataset assembly and zero forecasting
│   │   ├── models.py             # LightGBM/XGBoost training and prediction
│   │   └── pipeline.py           # End-to-end final pipeline
│   ├── features/
│   │   └── make_features.py      # Shared date, aggregate, lag/rolling features
│   ├── models/
│   │   └── train_lgbm.py         # Shared LightGBM helpers
│   └── utils/
│       └── metrics.py            # RMSLE metric
├── submissions/
│   └── ensemble_v1_full.csv      # Final local submission file, ignored by Git
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
python scripts/run_ensemble_v1.py --mode fast
```

Final full run:

```powershell
python scripts/run_ensemble_v1.py --mode full
```

The final submission is saved to:

```text
submissions/ensemble_v1_full.csv
```

## Notes

Raw data, generated outputs, and submissions are ignored by Git to avoid large-file upload problems. The final CSV is kept locally for Kaggle submission.

