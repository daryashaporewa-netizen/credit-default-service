"""
JSON-логгер запросов на инференс.

Принципы:
- по одной строке JSON на событие (JSONL) — стандартный формат для приёма
  в ELK/Loki;
- НЕ логируем фичи клиента: в реальном банке это PII (возраст, лимит,
  долговая нагрузка). В учебном проекте мы тоже соблюдаем этот принцип
  и явно указываем это в README;
- путь к лог-файлу можно переопределить через LOG_DIR (для Docker volume).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

PREDICTION_LOGGER_NAME = "credit_default.predictions"


class JsonFormatter(logging.Formatter):
    """Каждая запись logger.info(extra={...}) превращается в одну JSON-строку."""

    def format(self, record: logging.LogRecord) -> str:
        # extra-поля, которые мы умеем логировать
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in (
            "endpoint",
            "user_id",
            "model_version",
            "ab_group",
            "prediction",
            "probability_default",
            "request_id",
            "status",
            "error",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def get_prediction_logger(log_dir: Optional[Path] = None) -> logging.Logger:
    log_dir = Path(os.environ.get("LOG_DIR", log_dir or DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "predictions.log"

    logger = logging.getLogger(PREDICTION_LOGGER_NAME)
    if logger.handlers:
        # Уже сконфигурирован (например, перезагрузка Flask reloader'ом)
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    file_handler = RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(JsonFormatter())
    logger.addHandler(file_handler)

    # Дублируем в stdout — удобно при работе в Docker (docker logs)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())
    logger.addHandler(stream_handler)

    return logger


def log_prediction(
    logger: logging.Logger,
    endpoint: str,
    user_id: Optional[str],
    model_version: str,
    prediction: int,
    probability_default: float,
    ab_group: Optional[str] = None,
) -> None:
    """Удобная обёртка с фиксированным контрактом полей."""
    extra = {
        "endpoint": endpoint,
        "user_id": user_id,
        "model_version": model_version,
        "prediction": int(prediction),
        "probability_default": float(probability_default),
        "status": "ok",
    }
    if ab_group is not None:
        extra["ab_group"] = ab_group
    logger.info("prediction", extra=extra)
