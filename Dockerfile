FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN useradd --create-home --uid 10001 app \
    && mkdir /app \
    && chown app:app /app
WORKDIR /app
COPY requirements-server.txt ./
RUN python -m pip install --no-cache-dir -r requirements-server.txt
COPY --chown=app:app . .
USER app
RUN mkdir -p games logs messages

EXPOSE 5000
# Process-local state and locks require exactly one worker; do not use --preload.
CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5000", "--timeout", "120", "web_player:app"]