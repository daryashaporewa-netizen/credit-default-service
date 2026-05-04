"""
Подсчёт метрик качества для бинарного классификатора дефолта.

Используется как из train.py (после обучения), так и при желании
отдельно по уже сохранённой .joblib-модели.
"""

from __future__ import annotations

from typing import Dict

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


def compute_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Считаем тот набор, который требует задание.

    accuracy в задаче с дисбалансом классов малоинформативна; ключевые
    метрики для кредитного скоринга — recall и F1 для класса 1 (дефолт)
    и ROC-AUC как порого-независимая оценка.
    """
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "support_positive": int(np.sum(y_true == 1)),
        "support_negative": int(np.sum(y_true == 0)),
    }
