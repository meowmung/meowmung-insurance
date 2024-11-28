#!/bin/bash

echo "Initializing Airflow Database..."
airflow db init

echo "Starting Airflow Scheduler..."
airflow scheduler &

echo "Starting Airflow Webserver..."
airflow webserver --port 8080
