import sys
import os
from datetime import datetime
import pymysql

pymysql.install_as_MySQLdb()
from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.operators.python import PythonOperator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bots.query import insert_insurances, insert_terms, insert_results


default_args = {
    "owner": "lsjdg",
    "start_date": datetime(2024, 11, 25),
}

with DAG(
    dag_id="meowmung_insurancedata",
    schedule_interval="@monthly",
    default_args=default_args,
    tags=["meowmung", "insurance_data", "pet_health"],
    catchup=False,
) as dag:

    create_insurance_table = MySqlOperator(
        task_id="create_insurance_table",
        mysql_conn_id="meowmung_mysql",
        sql="""
        CREATE TABLE IF NOT EXISTS Insurance (
            insurance_id VARCHAR(25) PRIMARY KEY,
            company VARCHAR(25),
            insurance_item VARCHAR(50),
            logo VARCHAR(50)
        );
    """,
    )

    create_terms_table = MySqlOperator(
        task_id="create_terms_table",
        mysql_conn_id="meowmung_mysql",
        sql="""
        CREATE TABLE IF NOT EXISTS Terms (
            term_id VARCHAR(25) PRIMARY KEY,
            insurance_id VARCHAR(50),
            term_name VARCHAR(50),
            term_causes VARCHAR(255),
            term_limits VARCHAR(255),
            term_details VARCHAR(255),
            INDEX idx_insurance_id (insurance_id),
            FOREIGN KEY (insurance_id) REFERENCES Insurance(insurance_id)
        );
    """,
    )

    create_results_table = MySqlOperator(
        task_id="create_results_table",
        mysql_conn_id="meowmung_mysql",
        sql="""
        CREATE TABLE IF NOT EXISTS Results (
            result_id BIGINT PRIMARY KEY AUTO_INCREMENT,
            term_id VARCHAR(25),
            disease_name VARCHAR(25),
            UNIQUE (term_id, disease_name),
            INDEX idx_term_id (term_id),
            FOREIGN KEY (term_id) REFERENCES Terms(term_id)
        );
    """,
    )

    insert_insurances = PythonOperator(
        task_id="insert_insurances",
        python_callable=insert_insurances,
        op_kwargs={
            "dir_path": "/opt/data/summaries/*.json",
            "table_name": "Insurance",
        },
    )

    insert_terms = PythonOperator(
        task_id="insert_terms",
        python_callable=insert_terms,
        op_kwargs={
            "dir_path": "/opt/data/summaries/*.json",
            "table_name": "Terms",
        },
    )

    insert_results = PythonOperator(
        task_id="insert_results",
        python_callable=insert_results,
        op_kwargs={
            "dir_path": "/opt/data/summaries/*.json",
            "table_name": "Results",
        },
    )

    (
        create_insurance_table
        >> create_terms_table
        >> create_results_table
        >> insert_insurances
        >> insert_terms
        >> insert_results
    )
