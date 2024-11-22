from datetime import datetime
import json
from airflow import DAG

default_args = {"owner": "lsjdg", "start_date": datetime(2024, 11, 27)}

with DAG(
    dag_id="meowmung_pipeline",
    schedule_interval="@monthly",  # 일주일 단위는 @weekly
    default_args=default_args,
    tags=["meowmung", "user_data", "pet_health"],
    catchup=False,  # 과거의 스케줄된 run 들을 처리할지 여부, True 면 start_date 부터 모든 주기마다의 run 실행
) as dag:
    pass
