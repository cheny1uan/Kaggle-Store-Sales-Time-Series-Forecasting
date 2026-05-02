from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble_v2_plus.pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    result = run(mode=args.mode)
    print(f"Validation RMSLE: {result['validation_rmsle']:.5f}")
    print(f"Base validation RMSLE: {result['base_validation_rmsle']:.5f}")
    print(f"Calibration strategy: {result['calibration_strategy']}")
    print(f"Saved submission to {result['submission_path']}")


if __name__ == "__main__":
    main()
