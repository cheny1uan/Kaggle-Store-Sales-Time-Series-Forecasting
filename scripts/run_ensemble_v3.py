from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ensemble_v3.config import MODES
from src.ensemble_v3.pipeline import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), default="fast")
    args = parser.parse_args()

    result = run(mode=args.mode)
    print(f"Validation RMSLE: {result['validation_rmsle']:.5f}")
    print(f"Base validation RMSLE: {result['base_validation_rmsle']:.5f}")
    print(f"Transaction validation RMSLE: {result['tx_validation_rmsle']:.5f}")
    print(f"Calibration strategy: {result['calibration_strategy']}")
    print(f"Sales weights: {result['sales_weights']}")
    print(f"Transaction weights: {result['tx_weights']}")
    print(f"Saved submission to {result['submission_path']}")


if __name__ == "__main__":
    main()
