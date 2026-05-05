# Submission Queue

## Current Best Result

| File | Public Score | Note |
|---|---:|---|
| `submissions/final_submission.csv` | **0.39969** | Final submission |

## Historical Anchor

| File | Public Score | Note |
|---|---:|---|
| `submissions/blend_anchor.csv` | 0.40387 | Best result before the final blend |

## Reference Candidate

| File | Note |
|---|---|
| `submissions/reference_submission.csv` | Complementary reference used for the final log-space blend |

## Reproduce the Final File

```powershell
python scripts/blend_submissions.py `
  --base submissions/blend_anchor.csv `
  --plus submissions/reference_submission.csv `
  --weights 0.08 `
  --methods log `
  --output submissions/final_submission.csv
```

## Why This Blend Was Kept

- It keeps the strongest anchor result intact.
- It adds a small complementary correction in log space.
- It stays stable under RMSLE and does not overreact to outliers.
