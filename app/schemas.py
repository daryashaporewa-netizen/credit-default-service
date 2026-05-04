"""
Лёгкая ручная валидация входного JSON для /predict и /predict_ab.

FastAPI/pydantic не используем (запрещено заданием), достаточно проверить:
- что есть объект 'features';
- что в нём присутствуют все 23 ожидаемых ключа;
- что значения приводятся к числам.

Возвращаем нормализованный DataFrame (одна строка) или текст ошибки.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.preprocessing import EXPECTED_FEATURES, coerce_features

ALLOWED_MODEL_VERSIONS = {"v1", "v2"}


def parse_predict_payload(payload: Dict[str, Any]) -> Tuple[
    Optional[pd.DataFrame], Optional[str], Optional[str], Optional[str]
]:
    """Распарсить тело /predict.

    Returns:
        (features_df, user_id, model_version, error_message).
        Если error_message не None — остальные значения игнорировать.
    """
    if not isinstance(payload, dict):
        return None, None, None, "request body must be a JSON object"

    user_id = payload.get("user_id")
    if user_id is not None and not isinstance(user_id, str):
        return None, None, None, "user_id must be a string"

    model_version = payload.get("model_version", "v1")
    if model_version not in ALLOWED_MODEL_VERSIONS:
        return (
            None,
            None,
            None,
            f"model_version must be one of {sorted(ALLOWED_MODEL_VERSIONS)}",
        )

    features = payload.get("features")
    if not isinstance(features, dict):
        return None, None, None, "field 'features' is required and must be an object"

    missing = [c for c in EXPECTED_FEATURES if c not in features]
    if missing:
        return None, None, None, f"missing features: {missing}"

    try:
        df = coerce_features(features)
    except (TypeError, ValueError) as exc:
        return None, None, None, f"invalid feature value: {exc}"

    return df, user_id, model_version, None


def parse_predict_ab_payload(payload: Dict[str, Any]) -> Tuple[
    Optional[pd.DataFrame], Optional[str], Optional[str]
]:
    """Распарсить тело /predict_ab. user_id обязателен.

    Returns:
        (features_df, user_id, error_message).
    """
    if not isinstance(payload, dict):
        return None, None, "request body must be a JSON object"

    user_id = payload.get("user_id")
    if not isinstance(user_id, str) or not user_id:
        return None, None, "user_id is required for /predict_ab and must be a non-empty string"

    features = payload.get("features")
    if not isinstance(features, dict):
        return None, None, "field 'features' is required and must be an object"

    missing = [c for c in EXPECTED_FEATURES if c not in features]
    if missing:
        return None, None, f"missing features: {missing}"

    try:
        df = coerce_features(features)
    except (TypeError, ValueError) as exc:
        return None, None, f"invalid feature value: {exc}"

    return df, user_id, None
