from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score


def regression_report(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def classification_report(y_true, y_pred) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
    }


def cv_summary(scores: np.ndarray) -> tuple[float, float]:
    """Restituisce (mean, std) in float."""
    return float(np.mean(scores)), float(np.std(scores))


def pretty(d: dict) -> str:
    out = []
    for k, v in d.items():
        if isinstance(v, (int, float, np.floating)):
            out.append(f"{k}: {v:.4f}")
        else:
            out.append(f"{k}: {v}")
    return "\n".join(out)
