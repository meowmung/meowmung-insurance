from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import glob


default_args = {
    "owner": "lsjdg",
    "start_date": datetime(2024, 11, 25),
}


def summarize_dir():
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from bots.summary import save_summaries, extract_company_name

    file_paths = glob.glob(f"/opt/data/pdf/*.pdf")

    for path in file_paths:
        company = extract_company_name(path)
        save_summaries(company, "summary")
        print(f"Summary for {path} saved.")


with DAG(
    dag_id="meowmung_summary",
    schedule_interval="@monthly",
    default_args=default_args,
    tags=["meowmung", "summary", "pet_health"],
    catchup=False,
) as dag:

    summarize = PythonOperator(
        task_id="summarize",
        python_callable=summarize_dir,
    )
