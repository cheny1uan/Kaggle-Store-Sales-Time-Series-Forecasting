from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hybrid_v2.io import save_submission
from src.hybrid_v2.pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    result = run(mode=args.mode)
    print(f"Validation RMSLE: {result['validation_rmsle']:.5f}")
    out_path = Path(ROOT / "submissions" / f"hybrid_v2_{args.mode}.csv")
    save_submission(result["submission"], out_path)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    main()
