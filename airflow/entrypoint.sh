#!/bin/bash

echo "Initializing Airflow Database..."
airflow db migrate && airflow connections create-default-connections

echo "Starting Airflow Scheduler..."
airflow scheduler &

echo "Starting Airflow Webserver..."
airflow webserver --port 8080
