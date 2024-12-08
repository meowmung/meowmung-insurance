#!/bin/bash

echo "Initializing Airflow Database..."
airflow db init && airflow connections create-default-connections

echo "Creating Admin User..."
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

echo "Starting Airflow Scheduler..."
airflow scheduler &

echo "Waiting for Airflow Scheduler to start..."
sleep 10 

echo "Starting Airflow Webserver..."
airflow webserver --port 8080 &

echo "Creating MySQL connector..."
python connector.py

echo "Airflow services are running. Logs will be shown below."

tail -f /dev/null
