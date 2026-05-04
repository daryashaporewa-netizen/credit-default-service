# Credit Default Prediction Service

Production-like ML-сервис для прогнозирования дефолта по кредитной карте.  
Покрывает полный цикл: обучение → сериализация → API → Docker → логирование → A/B-тест.

---

## 1. Что делает сервис

Flask API возвращает вероятность дефолта клиента.

Поддерживает 2 модели:
- `v1` — LogisticRegression (контроль)
- `v2` — GradientBoosting (тест)

A/B реализован через `/predict_ab`:
- сплит 50/50 по `user_id`
- детерминированный (один пользователь всегда в одной группе)

---

## 2. Датасет

UCI Credit Card Default Dataset  
30 000 строк, 23 признака, target — `default.payment.next.month`  
Дисбаланс: ~22% дефолтов

Файл нужно положить в:
```
data/raw/UCI_Credit_Card.csv
```

---

## 3. Структура проекта

```
ml-credit-default-service/
├── app/
├── src/
├── models/
├── data/raw/
├── logs/
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 4. Установка

```
bash
git clone https://github.com/daryashaporewa-netizen/credit-default-service.git
cd ml-credit-default-service

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 5. Обучение

```bash 
python src/train.py
```

Создаст:

```
models/model_v1.joblib
models/model_v2.joblib
models/metrics.json
```

Метрики (test):

| model | F1    | Recall | Precision | ROC-AUC |
|------ |----   |--------|---------- |-------- |
| v1    | 0.465 | 0.63   | 0.37      | 0.71    |
| v2    | 0.467 | 0.36   | 0.66      | 0.78    |

---

## 6. Запуск API

```bash 
gunicorn -b 0.0.0.0:5001 app.api:app
```
---

## 7. Docker

```bash
docker build -t credit-default-service .
docker run -p 5001:5000 credit-default-service

На macOS порт 5000 был занят → использован 5001
```
---

## 8. Docker Compose

```bash
docker compose up --build
```
---

## 9. API

### Health

```bash
curl http://localhost:5001/health
```

### Predict

```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_version": "v1",
    "features": {
      "LIMIT_BAL": 20000,
      "SEX": 2,
      "EDUCATION": 2,
      "MARRIAGE": 1,
      "AGE": 24,
      "PAY_0": 2,
      "PAY_2": 2,
      "PAY_3": -1,
      "PAY_4": -1,
      "PAY_5": -2,
      "PAY_6": -2,
      "BILL_AMT1": 3913,
      "BILL_AMT2": 3102,
      "BILL_AMT3": 689,
      "BILL_AMT4": 0,
      "BILL_AMT5": 0,
      "BILL_AMT6": 0,
      "PAY_AMT1": 0,
      "PAY_AMT2": 689,
      "PAY_AMT3": 0,
      "PAY_AMT4": 0,
      "PAY_AMT5": 0,
      "PAY_AMT6": 0
    }
  }'
```

Ответ:

```json
{
  "prediction": 1,
  "probability_default": 0.78,
  "model_version": "v1"
} 
```

### A/B Predict

```bash
curl -X POST http://localhost:5001/predict_ab \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "client_001",
    "features": {}
  }'
```

Ответ:

```json
{
  "prediction": 1,
  "probability_default": 0.75,
  "model_version": "v2",
  "ab_group": "B"
}
```

---

## 10. Формат входа

```json
{
  "model_version": "v1",
  "user_id": "client_001",
  "features": {23 поля}
}
```
---

## 11. Тесты

```bash
python -m pytest -q```

Результат:
```9 passed
```
---

## 12. Логирование

- формат: JSONL  
- файл: `logs/predictions.log`  
- PII (features) не логируются  

---

## 13. A/B тест

- split: 50/50  
- метрики: F1, Recall  
- бизнес: Approval Rate, Expected Loss  
- статтесты: z-test, t-test  

---

## 14. Архитектура

Монолитный Flask-сервис в Docker.

В `docs/ARCHITECTURE.md` описано:
- переход к микросервисам  
- RabbitMQ  
- ELK  
- MLflow  
- DVC  
- ONNX  

---

## 15. Docker Hub

```bash
docker tag credit-default-service <username>/credit-default-service
docker push <username>/credit-default-service

Docker Hub image: https://hub.docker.com/r/dariavdovina/credit-default-service
```

---

## Итог

Сервис:
- обучает модели  
- отдаёт предсказания  
- поддерживает A/B  
- контейнеризован  
- покрыт тестами  
