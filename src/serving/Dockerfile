FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
COPY src/serving ./src/serving

RUN pip install uv
RUN uv sync --frozen

EXPOSE 8080

ENV MODEL_PATH=/app/artifacts/model.pkl

CMD ["uv", "run", "uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8080"]
