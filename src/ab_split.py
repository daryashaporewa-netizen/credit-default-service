"""
Стабильное распределение клиента в A/B-группу по user_id.

Свойства, которые нам нужны:
- детерминированность: один и тот же user_id всегда попадает в одну группу,
  иначе сравнение моделей превратится в кашу;
- независимость от платформы: hash() в Python несолёный, между процессами
  разный, поэтому используем md5 — он стабилен;
- сплит ровно 50/50 в среднем на больших выборках.
"""

from __future__ import annotations

import hashlib
from typing import Literal

ABGroup = Literal["A", "B"]


def assign_ab_group(user_id: str) -> ABGroup:
    """Вернуть 'A' или 'B' детерминированно по user_id."""
    if not isinstance(user_id, str) or not user_id:
        raise ValueError("user_id must be a non-empty string for A/B assignment")
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    # Берём последний байт, чётный -> A, нечётный -> B.
    # На равномерном md5 это даёт сплит ~50/50.
    last_byte = int(digest[-2:], 16)
    return "A" if last_byte % 2 == 0 else "B"


def model_version_for_group(group: ABGroup) -> str:
    """Контракт A/B: A -> v1 (контроль), B -> v2 (тест)."""
    return "v1" if group == "A" else "v2"
