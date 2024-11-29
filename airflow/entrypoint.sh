#!/bin/bash

echo "Initializing Airflow Database..."
airflow db migrate && airflow connections create-default-connections

echo "Starting Airflow Scheduler..."
airflow scheduler &

echo "Starting Airflow Webserver..."
airflow webserver --port 8080 &

sleep 10

echo "Triggering DAGs..."
airflow dags trigger meowmung_insurancedata
airflow dags trigger meowmung_summary

tail -f /dev/null
