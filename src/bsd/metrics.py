"""Metrics shared by the baselines and the transformer, so every model is scored the same way.

"translated" (label 1) is the positive class, matching the paper's tables. Per-class recall is
reported too because the paper's main failure mode was synthetic recall collapsing out of domain.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute(y_true, y_pred, y_score=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_synthetic": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_translated": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if y_score is not None and len(set(y_true.tolist())) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def save_scores(model_name: str, split: str, y_true, y_score, scores_dir: Path = Path("results/scores")) -> None:
    """Keep per-row scores so later analysis (e.g. bsd.calibrate) needs no retraining."""
    scores_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": np.asarray(y_true), "score": np.asarray(y_score)}).to_parquet(
        scores_dir / f"{model_name}__{split}.parquet", index=False
    )
