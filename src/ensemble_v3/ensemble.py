from __future__ import annotations

import numpy as np


def inverse_score_weights(scores: dict[str, float]) -> dict[str, float]:
    raw = {name: 1.0 / max(score, 1e-9) for name, score in scores.items()}
    total = sum(raw.values())
    return {name: value / total for name, value in raw.items()}


def blend_predictions(predictions: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    pred = None
    for name, values in predictions.items():
        part = np.asarray(values, dtype=float) * weights[name]
        pred = part if pred is None else pred + part
    return np.clip(pred, 0, None)
