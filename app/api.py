"""
Flask API для сервиса прогнозирования дефолта по кредитным картам.

Endpoints:
- GET  /health        — статус сервиса и факт загрузки моделей.
- POST /predict       — инференс по выбранной версии модели (v1 по умолчанию).
- POST /predict_ab    — инференс с A/B-распределением по user_id.

Дизайн:
- модели грузятся ОДИН раз при старте процесса (model_loader.registry.load());
- внутри запросов мы только predict-им, никаких .fit();
- каждый успешный инференс пишется в JSONL-лог (logs/predictions.log);
- персональные данные клиента в лог не пишутся.

Запуск (dev):
    python -m flask --app app.api run --host=0.0.0.0 --port=5000
Запуск (prod-like, как в Docker):
    gunicorn -b 0.0.0.0:5000 app.api:app
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from flask import Flask, jsonify, request

# Делаем app/api.py запускаемым и из корня репо, и из любой CWD
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.logger import get_prediction_logger, log_prediction  # noqa: E402
from app.model_loader import registry  # noqa: E402
from app.schemas import parse_predict_ab_payload, parse_predict_payload  # noqa: E402
from src.ab_split import assign_ab_group, model_version_for_group  # noqa: E402


def create_app() -> Flask:
    app = Flask(__name__)

    # Загружаем модели один раз. Если не получилось — поднимаем исключение,
    # пусть процесс падает; так в Docker это сразу видно по docker logs.
    if not registry.is_loaded:
        registry.load()

    prediction_logger = get_prediction_logger()

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "credit-default-prediction",
                "models_loaded": registry.is_loaded,
                "model_versions": registry.versions,
            }
        )

    @app.post("/predict")
    def predict():
        request_id = str(uuid.uuid4())
        payload = request.get_json(silent=True)
        df, user_id, model_version, err = parse_predict_payload(payload or {})
        if err:
            return jsonify({"error": err, "request_id": request_id}), 400

        try:
            model = registry.get(model_version)
            proba = float(model.predict_proba(df)[0, 1])
            pred = int(proba >= 0.5)
        except Exception as exc:  # pragma: no cover - неожиданные ошибки модели
            return (
                jsonify({"error": f"inference failed: {exc}", "request_id": request_id}),
                500,
            )

        log_prediction(
            prediction_logger,
            endpoint="/predict",
            user_id=user_id,
            model_version=model_version,
            prediction=pred,
            probability_default=proba,
        )
        return jsonify(
            {
                "prediction": pred,
                "probability_default": round(proba, 4),
                "model_version": model_version,
                "request_id": request_id,
            }
        )

    @app.post("/predict_ab")
    def predict_ab():
        request_id = str(uuid.uuid4())
        payload = request.get_json(silent=True)
        df, user_id, err = parse_predict_ab_payload(payload or {})
        if err:
            return jsonify({"error": err, "request_id": request_id}), 400

        ab_group = assign_ab_group(user_id)
        model_version = model_version_for_group(ab_group)

        try:
            model = registry.get(model_version)
            proba = float(model.predict_proba(df)[0, 1])
            pred = int(proba >= 0.5)
        except Exception as exc:  # pragma: no cover
            return (
                jsonify({"error": f"inference failed: {exc}", "request_id": request_id}),
                500,
            )

        log_prediction(
            prediction_logger,
            endpoint="/predict_ab",
            user_id=user_id,
            model_version=model_version,
            prediction=pred,
            probability_default=proba,
            ab_group=ab_group,
        )
        return jsonify(
            {
                "prediction": pred,
                "probability_default": round(proba, 4),
                "model_version": model_version,
                "ab_group": ab_group,
                "request_id": request_id,
            }
        )

    @app.errorhandler(404)
    def not_found(_e):
        return jsonify({"error": "endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(_e):
        return jsonify({"error": "method not allowed"}), 405

    return app


# Объект приложения, который видит и `flask run`, и gunicorn.
app = create_app()


if __name__ == "__main__":
    # Удобный fallback: `python app/api.py`
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
