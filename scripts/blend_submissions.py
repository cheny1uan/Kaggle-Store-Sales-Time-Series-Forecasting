from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _weight_label(weight: float) -> tuple[str, str]:
    plus_pct = weight * 100
    base_pct = (1.0 - weight) * 100

    def fmt(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.1f}".rstrip("0").rstrip(".").replace(".", "p")

    return fmt(plus_pct), fmt(base_pct)


def _validate_pair(base: pd.DataFrame, plus: pd.DataFrame) -> None:
    if list(base.columns) != ["id", "sales"] or list(plus.columns) != ["id", "sales"]:
        raise ValueError("Both submission files must have columns: id,sales")
    if len(base) != len(plus) or not base["id"].equals(plus["id"]):
        raise ValueError("Submission ids do not match.")


def _stats(out: pd.DataFrame) -> dict[str, float]:
    return {
        "rows": float(len(out)),
        "min_sales": float(out["sales"].min()),
        "max_sales": float(out["sales"].max()),
        "mean_sales": float(out["sales"].mean()),
    }


def blend_pair(
    base_path: Path,
    plus_path: Path,
    plus_weight: float,
    output_path: Path,
    method: str = "linear",
) -> dict[str, float]:
    base = pd.read_csv(base_path)
    plus = pd.read_csv(plus_path)

    _validate_pair(base, plus)

    base_weight = 1.0 - plus_weight
    out = base.copy()
    base_sales = base["sales"].astype(float).clip(lower=0)
    plus_sales = plus["sales"].astype(float).clip(lower=0)

    if method == "linear":
        out["sales"] = plus_weight * plus_sales + base_weight * base_sales
    elif method == "log":
        out["sales"] = np.expm1(
            plus_weight * np.log1p(plus_sales)
            + base_weight * np.log1p(base_sales)
        )
    elif method == "sqrt":
        out["sales"] = (
            plus_weight * np.sqrt(plus_sales)
            + base_weight * np.sqrt(base_sales)
        ) ** 2
    else:
        raise ValueError(f"Unknown blend method: {method}")

    out["sales"] = np.clip(out["sales"].to_numpy(), 0, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return _stats(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/blend_anchor.csv")
    parser.add_argument("--plus", default="submissions/reference_submission.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output path for a single blend result.",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=[0.55, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80],
        help="Weights assigned to the --plus submission; base weight is 1 - weight.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["linear", "log", "sqrt"],
        default=["linear", "log", "sqrt"],
    )
    args = parser.parse_args()

    base_path = ROOT / args.base
    plus_path = ROOT / args.plus
    if args.output is not None:
        if len(args.weights) != 1 or len(args.methods) != 1:
            raise ValueError("--output can only be used with exactly one weight and one method.")
        weight = args.weights[0]
        method = args.methods[0]
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = ROOT / output_path
        stats = blend_pair(base_path, plus_path, weight, output_path, method=method)
        print(
            f"saved {output_path.name}: "
            f"rows={int(stats['rows'])}, "
            f"min={stats['min_sales']:.4f}, "
            f"max={stats['max_sales']:.4f}, "
            f"mean={stats['mean_sales']:.4f}"
        )
        return

    for method in args.methods:
        for weight in args.weights:
            plus_pct, base_pct = _weight_label(weight)
            prefix = "blend" if method == "linear" else f"{method}blend"
            out_path = ROOT / "submissions" / f"{prefix}_{plus_pct}plus_{base_pct}v2.csv"
            stats = blend_pair(base_path, plus_path, weight, out_path, method=method)
            print(
                f"saved {out_path.name}: "
                f"rows={int(stats['rows'])}, "
                f"min={stats['min_sales']:.4f}, "
                f"max={stats['max_sales']:.4f}, "
                f"mean={stats['mean_sales']:.4f}"
            )


if __name__ == "__main__":
    main()
