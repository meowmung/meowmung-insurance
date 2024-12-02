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

echo "Starting Airflow Webserver..."
airflow webserver --port 8080 &

# sleep 10

# echo "Triggering DAGs..."
# airflow dags trigger meowmung_insurancedata
# airflow dags trigger meowmung_summary
# airflow dags trigger meowming_trainer

tail -f /dev/null
