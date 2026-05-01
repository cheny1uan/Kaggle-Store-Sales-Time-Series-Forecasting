from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble_v1.pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    result = run(mode=args.mode)
    print(f"Validation RMSLE: {result['validation_rmsle']:.5f}")
    print(f"Saved submission to {result['submission_path']}")


if __name__ == "__main__":
    main()

