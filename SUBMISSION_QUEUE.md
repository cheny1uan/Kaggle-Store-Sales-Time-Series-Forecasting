# Submission Queue

## Current Best Result

| File | Public Score | Rank | Note |
|---|---:|---:|---|
| `submissions/v4_logblend_55since_45full.csv` | **0.40387** | **100** | Final submission |
| `submissions/ensemble_v4_since2015.csv` | 0.38540 | - | Best single v4 window model on local validation |
| `submissions/ensemble_v4_full.csv` | 0.38599 | - | Full-history backup |

## Current Advice

If you need a safe submission, use:

```text
submissions/v4_logblend_55since_45full.csv
```

If you want a conservative backup, use:

```text
submissions/ensemble_v4_since2015.csv
```

## Why This Is the Final Choice

- `v4_since2015` captures the more recent distribution.
- `v4_full` keeps longer historical context.
- A log-space blend is more aligned with the RMSLE metric than a raw linear average.

