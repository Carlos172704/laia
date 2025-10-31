set -e

mlflow server \
  --host 0.0.0.0 \
  --port 5050 \
  --backend-store-uri sqlite:///mlflow/mlflow.db \
  --artifacts-destination /mlflow/mlruns \
  --serve-artifacts \
  --allowed-hosts '*'
