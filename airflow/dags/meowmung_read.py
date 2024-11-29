from airflow import DAG
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator
import pandas as pd
from datetime import datetime


def fetch_data_from_mysql(**kwargs):
    mysql_hook = MySqlHook(mysql_conn_id="MEOWMUNG")

    query = "SELECT * FROM traindata_dog;"
    df = mysql_hook.get_pandas_df(query)

    kwargs["ti"].xcom_push(key="traindata_dog_df", value=df.to_json())


default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 11, 26),
}

with DAG(
    "read_data_from_mysql",
    default_args=default_args,
    schedule="@daily",
    tags=["meowmung", "insurance_data", "dog_health"],
) as dag:

    fetch_data = PythonOperator(
        task_id="fetch_data_from_mysql",
        python_callable=fetch_data_from_mysql,
    )

    fetch_data
