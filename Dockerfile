# Базовый образ — slim, чтобы итоговый артефакт был ~200MB,
# а не 1GB как с полным python-образом.
FROM python:3.11-slim

# Не пишем .pyc, не буферизуем stdout — так удобнее в docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Создаём непривилегированного пользователя — не даём root внутри контейнера.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Сначала зависимости — слой кэшируется и пересобирается только при
# изменении requirements.txt (Docker layer caching).
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Затем код приложения и обученные артефакты.
COPY app /app/app
COPY src /app/src
COPY models /app/models

# Папка для логов будет монтироваться как volume.
RUN mkdir -p /app/logs && chown -R app:app /app

USER app

EXPOSE 5000

# Production-ready запуск через gunicorn (см. docs/ARCHITECTURE.md).
# 2 воркера — для учебного проекта достаточно; в production обычно
# 2*CPU+1 и таймауты подкручиваются под профиль нагрузки.
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "app.api:app"]
