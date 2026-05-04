"""
Загрузка обученных моделей в память один раз при старте процесса.

Почему так:
- inference в Flask должен быть дешёвым: распаковывать joblib на каждом
  запросе нельзя;
- две модели грузим параллельно как v1/v2 для эндпоинтов /predict
  и /predict_ab.

Путь к моделям можно переопределить переменной окружения MODELS_DIR —
это пригодится в Docker, где код и модели монтируются в /app.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

log = logging.getLogger(__name__)


class ModelRegistry:
    """Простой in-memory реестр моделей версий v1/v2."""

    def __init__(self, models_dir: Optional[Path] = None) -> None:
        self.models_dir = Path(
            os.environ.get("MODELS_DIR", models_dir or DEFAULT_MODELS_DIR)
        )
        self._models: Dict[str, object] = {}

    def load(self) -> None:
        v1_path = self.models_dir / "model_v1.joblib"
        v2_path = self.models_dir / "model_v2.joblib"
        if not v1_path.exists() or not v2_path.exists():
            raise FileNotFoundError(
                f"Models not found in {self.models_dir}. "
                "Сначала запустите `python src/train.py`."
            )
        self._models["v1"] = joblib.load(v1_path)
        self._models["v2"] = joblib.load(v2_path)
        log.info("models loaded: v1=%s, v2=%s", v1_path, v2_path)

    def get(self, version: str):
        if version not in self._models:
            raise KeyError(
                f"Unknown model_version='{version}'. Available: {list(self._models)}"
            )
        return self._models[version]

    @property
    def is_loaded(self) -> bool:
        return bool(self._models)

    @property
    def versions(self):
        return list(self._models.keys())


# Singleton, импортируется из app.api
registry = ModelRegistry()
