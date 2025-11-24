FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./

RUN pip install uv
RUN uv sync

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8080

ENV MODEL_PATH=/app/artifacts/model.pkl

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
