from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def blend_pair(base_path: Path, plus_path: Path, plus_weight: float, output_path: Path) -> dict[str, float]:
    base = pd.read_csv(base_path)
    plus = pd.read_csv(plus_path)

    if list(base.columns) != ["id", "sales"] or list(plus.columns) != ["id", "sales"]:
        raise ValueError("Both submission files must have columns: id,sales")
    if len(base) != len(plus) or not base["id"].equals(plus["id"]):
        raise ValueError("Submission ids do not match.")

    base_weight = 1.0 - plus_weight
    out = base.copy()
    out["sales"] = (
        plus_weight * plus["sales"].astype(float)
        + base_weight * base["sales"].astype(float)
    )
    out["sales"] = np.clip(out["sales"].to_numpy(), 0, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return {
        "rows": float(len(out)),
        "min_sales": float(out["sales"].min()),
        "max_sales": float(out["sales"].max()),
        "mean_sales": float(out["sales"].mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/ensemble_v2_full.csv")
    parser.add_argument("--plus", default="submissions/ensemble_v2_plus_full.csv")
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.50, 0.70, 0.85],
        help="Weights assigned to ensemble_v2_plus; base weight is 1 - weight.",
    )
    args = parser.parse_args()

    base_path = ROOT / args.base
    plus_path = ROOT / args.plus
    for weight in args.weights:
        plus_pct = int(round(weight * 100))
        base_pct = int(round((1.0 - weight) * 100))
        out_path = ROOT / "submissions" / f"blend_{plus_pct}plus_{base_pct}v2.csv"
        stats = blend_pair(base_path, plus_path, weight, out_path)
        print(
            f"saved {out_path.name}: "
            f"rows={int(stats['rows'])}, "
            f"min={stats['min_sales']:.4f}, "
            f"max={stats['max_sales']:.4f}, "
            f"mean={stats['mean_sales']:.4f}"
        )


if __name__ == "__main__":
    main()
