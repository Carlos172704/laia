FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
COPY src/training ./src/training

RUN pip install uv
RUN uv sync --frozen

CMD ["uv", "run", "python", "src/training/train.py"]
