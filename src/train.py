"""
Точка входа для обучения двух моделей и сохранения артефактов.

Запуск:
    python -m src.train
    или
    python src/train.py

Что делает:
1. Загружает data/raw/UCI_Credit_Card.csv.
2. Удаляет ID, разделяет X/y, чистит EDUCATION/MARRIAGE.
3. Стратифицированный train/test split (test_size=0.2, random_state=42).
4. Собирает два Pipeline: preprocessor + LogisticRegression (v1)
   и preprocessor + GradientBoostingClassifier (v2).
5. Обучает обе модели.
6. Считает метрики на test (accuracy/precision/recall/F1/ROC-AUC + CM).
7. Сохраняет models/model_v1.joblib, models/model_v2.joblib, models/metrics.json.

Метрики не выдумываются — все значения считаются на отложенной выборке.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Делаем модуль запускаемым как `python src/train.py` и как `python -m src.train`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import compute_metrics  # noqa: E402
from src.preprocessing import build_preprocessor, split_features_target  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def build_model_v1() -> Pipeline:
    """Контрольная модель: логрег с балансировкой классов.

    class_weight='balanced' — компенсация дисбаланса (≈22% дефолтов),
    важна именно для recall на классе 1.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_model_v2() -> Pipeline:
    """Тестовая модель: GradientBoostingClassifier.

    Более ёмкая модель, обычно даёт лучший ROC-AUC и F1 на этом датасете.
    Без подбора гиперпараметров — учебный проект, важна реальность чисел,
    а не Kaggle-уровень.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_eval(model: Pipeline, X_train, y_train, X_test, y_test, name: str):
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["fit_time_seconds"] = round(fit_time, 3)
    metrics["model"] = name
    return model, metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Скачайте UCI_Credit_Card.csv и положите в data/raw/."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] loading {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"[train] dataset shape: {df.shape}")

    X, y = split_features_target(df)
    print(f"[train] features: {X.shape}, target balance: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"[train] train: {X_train.shape}, test: {X_test.shape}")

    print("[train] fitting model_v1 (LogisticRegression, balanced)...")
    model_v1, metrics_v1 = train_and_eval(
        build_model_v1(), X_train, y_train, X_test, y_test, "v1_logreg"
    )
    print("[train] v1 metrics:", json.dumps(metrics_v1, indent=2))

    print("[train] fitting model_v2 (GradientBoostingClassifier)...")
    model_v2, metrics_v2 = train_and_eval(
        build_model_v2(), X_train, y_train, X_test, y_test, "v2_gbm"
    )
    print("[train] v2 metrics:", json.dumps(metrics_v2, indent=2))

    v1_path = MODELS_DIR / "model_v1.joblib"
    v2_path = MODELS_DIR / "model_v2.joblib"
    joblib.dump(model_v1, v1_path)
    joblib.dump(model_v2, v2_path)
    print(f"[train] saved {v1_path}")
    print(f"[train] saved {v2_path}")

    metrics_payload = {
        "dataset": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "n_rows": int(df.shape[0]),
            "n_features": int(X.shape[1]),
            "positive_rate": float(y.mean()),
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
        },
        "model_v1": metrics_v1,
        "model_v2": metrics_v2,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"[train] saved {metrics_path}")


if __name__ == "__main__":
    main()
