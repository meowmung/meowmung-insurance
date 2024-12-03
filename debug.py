import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MODEL_NAME = f"best_clf_dog"
MLFLOW_TRACKING_URI = "localhost:5000"

# model_uri = "models:/best_clf_dog/1"
# model = mlflow.pyfunc.load_model(model_uri)

client = MlflowClient()
model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
