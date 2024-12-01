from airflow import DAG
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator
import pandas as pd
from datetime import datetime
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def fetch_data_from_mysql(**kwargs):
    mysql_hook = MySqlHook(mysql_conn_id="meowmung_mysql")
    query = "SELECT * FROM traindata_dog;"
    df = mysql_hook.get_pandas_df(query)

    if df.empty:
        raise ValueError("No data found in MySQL")

    kwargs["ti"].xcom_push(key="traindata_dog_df", value=df.to_json())


def model_training_and_tuning(ti, **kwargs):
    json_data = ti.xcom_pull(task_ids="fetch_data_from_mysql", key="traindata_dog_df")
    if json_data is None:
        raise ValueError("No data found in XCom")

    df = pd.read_json(json_data)
    X = df.drop("disease_code", axis=1)
    y = df["disease_code"]

    rf = RandomForestClassifier()
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    y_pred = grid_search.predict(X)
    report = classification_report(y, y_pred)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="weighted")
    recall = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    model_filename = "/tmp/best_model.pkl"
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
        },
    )


def push_model_to_mlflow(ti, **kwargs):
    model_filename = ti.xcom_pull(task_ids="train_and_tune_model", key="model_path")

    if model_filename is None:
        raise ValueError("No model path found in XCom")

    with open(model_filename, "rb") as f:
        best_model = pickle.load(f)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Model Training")

    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "best_model")

        metrics = ti.xcom_pull(task_ids="train_and_tune_model", key="metrics")
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
        )

        mlflow.log_param("classification_report", metrics["classification_report"])


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 11, 26),
    "retries": 1,
}

with DAG(
    "meowmung_trainer",
    default_args=default_args,
    schedule="@monthly",
    tags=["meowmung", "insurance_data", "pet_health"],
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

    fetch_data >> train_model >> push_mlflow
