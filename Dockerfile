FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar código e modelo
COPY src ./src
COPY artifacts ./artifacts

EXPOSE 8080

ENV MODEL_PATH=/app/artifacts/model.pkl

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
