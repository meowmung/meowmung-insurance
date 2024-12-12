from airflow import DAG
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator
import pymysql

pymysql.install_as_MySQLdb()
import pandas as pd
import os
from datetime import datetime
import pickle
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from dotenv import load_dotenv

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") + ":5000"


def fetch_data_from_mysql(**kwargs):
    mysql_hook = MySqlHook(mysql_conn_id="meowmung_mysql")
    query = "SELECT * FROM traindata_cat;"
    df = mysql_hook.get_pandas_df(query)
    print(df["age"].unique())

    if df.empty:
        raise ValueError("No data found in MySQL")

    kwargs["ti"].xcom_push(key="traindata_cat_df", value=df.to_json())


def model_training_and_tuning(ti, **kwargs):
    json_data = ti.xcom_pull(task_ids="fetch_data_from_mysql", key="traindata_cat_df")
    if json_data is None:
        raise ValueError("No data found in XCom")

    df = pd.read_json(json_data)
    X = df.drop(["train_id", "disease_code"], axis=1)
    y = df["disease_code"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    xgb = XGBClassifier(
        random_state=42, use_label_encoder=False, eval_metric="mlogloss"
    )
    param_grid = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 6, 10],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.2],
        "min_child_weight": [1, 2],
    }

    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = grid_search.predict(X_test)
    report = classification_report(y_test, y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)
    threat_score = precision + recall + f1

    model_filename = "/tmp/best_model_cat.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(best_model, f)

    ti.xcom_push(key="model_path", value=model_filename)
    ti.xcom_push(
        key="metrics",
        value={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
            "matthews_corrcoef": mcc,
            "threat_score": threat_score,
        },
    )


def push_model_to_mlflow(ti, **kwargs):
    model_filename = ti.xcom_pull(task_ids="train_and_tune_model", key="model_path")

    if model_filename is None:
        raise ValueError("No model path found in XCom")

    with open(model_filename, "rb") as f:
        best_model = pickle.load(f)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Meowmung Dog Trainer")

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(best_model, "best_clf_cat")

        metrics = ti.xcom_pull(task_ids="train_and_tune_model", key="metrics")
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "matthews_corrcoef": metrics["matthews_corrcoef"],
                "threat_score": metrics["threat_score"],
            }
        )

        mlflow.log_param("classification_report", metrics["classification_report"])

        model_uri = f"runs:/{run.info.run_id}/best_clf_cat"
        registered_model_name = "best_clf_cat"

        client = MlflowClient()
        try:
            client.create_registered_model(registered_model_name)
        except Exception:
            print(f"Model '{registered_model_name}' already exists in the registry.")

        model_version = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=run.info.run_id,
        )
        print(f"Registered model version: {model_version.version}")

        client.transition_model_version_stage(
            name=registered_model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"Model version {model_version.version} transitioned to 'Production'")


def save_model(pet_type, MODEL_STAGE, metric_name, ascending=True, **kwargs):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        MODEL_NAME = f"best_clf_{pet_type}"
        client = MlflowClient()

        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        if not model_versions:
            raise ValueError(f"No registered models found for {MODEL_NAME}")

        production_models = [
            mv for mv in model_versions if mv.current_stage == MODEL_STAGE
        ]

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

        local_model_path = f"{os.getenv("S3_URI")}data/models/best_clf_{pet_type}.pkl"
        with open(local_model_path, "wb") as f:
            pickle.dump(model, f)

        print(f"Model saved locally at: {local_model_path}")

    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        raise


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 11, 26),
    "retries": 1,
}

with DAG(
    "meowmung_cat_trainer",
    default_args=default_args,
    schedule="@monthly",
    tags=["meowmung", "pet_health"],
) as dag:

    fetch_data = PythonOperator(
        task_id="fetch_data_from_mysql",
        python_callable=fetch_data_from_mysql,
    )

    train_model = PythonOperator(
        task_id="train_and_tune_model",
        python_callable=model_training_and_tuning,
    )

    push_mlflow = PythonOperator(
        task_id="push_model_to_mlflow",
        python_callable=push_model_to_mlflow,
    )

    save_model = PythonOperator(task_id="save_model", python_callable=save_model)

    fetch_data >> train_model >> push_mlflow >> save_model
