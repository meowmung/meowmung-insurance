import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_STAGE = "Production"


def load_best_model(pet_type, MODEL_STAGE, metric_name, ascending=True):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MODEL_NAME = f"best_clf_{pet_type}"
    client = MlflowClient()

    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

    if not model_versions:
        raise ValueError(f"No registered models found for {MODEL_NAME}")

    production_models = [mv for mv in model_versions if mv.current_stage == MODEL_STAGE]

    if not production_models:
        raise ValueError(f"No models in stage '{MODEL_STAGE}' for {MODEL_NAME}")

    model_metrics = []
    for model_version in production_models:
        run_id = model_version.run_id
        run_data = client.get_run(run_id).data
        metric_value = run_data.metrics.get(metric_name)
        if metric_value is not None:
            model_metrics.append((run_id, metric_value))

    if not model_metrics:
        raise ValueError(f"No metrics found for models in stage '{MODEL_STAGE}'")

    model_metrics.sort(key=lambda x: x[1], reverse=not ascending)
    best_run_id = model_metrics[0][0]

    model_uri = f"runs:/{best_run_id}/best_clf_{pet_type}"
    model = mlflow.sklearn.load_model(model_uri)

    print(f"Model (run_id: {best_run_id}) loaded successfully.")
    return model


def pred_ill(pet_type, age, gender, breed, weight, food_count, neutered):

    model = load_best_model(
        pet_type, MODEL_STAGE, metric_name="accuracy", ascending=True
    )

    X = pd.DataFrame(
        [
            {
                "age": age,
                "weight": weight,
                "food_count": food_count,
                "breed_code": breed,
                "gender": gender,
                "neutered": neutered,
            }
        ]
    )
    predicted = model.predict(X)[0]

    predicted_code = int(predicted)

    # if pet_type == "dog":
    #     if predicted_code in [0, 1, 2]:
    #         return predicted_code
    #     elif predicted_code == 3:
    #         return 5
    #     elif predicted_code == 4:
    #         return 3

    # if pet_type == "cat":
    #     if predicted_code == 0:
    #         return 0
    #     return predicted_code + 3

    return predicted_code


print(pred_ill("dog", 9, 0, 21, 4.5, 1, 1))
