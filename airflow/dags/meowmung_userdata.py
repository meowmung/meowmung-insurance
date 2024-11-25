from airflow import DAG
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import pickle
from dotenv import load_dotenv
from sqlalchemy import create_engine
from airflow.providers.mysql.hooks.mysql import MySqlHook

default_args = {
    "owner": "lsjdg",
    "start_date": datetime(2024, 11, 25),
}

CSV_DIR = "data/csv"
MODEL_PATH = "/absolute/path/to/models/model.pkl"


def insert_csv(**kwargs):
    today = datetime.today().strftime("%Y_%m")
    CSV_PATH = os.path.join(CSV_DIR, f"userdata_{today}.csv")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    new_data = pd.read_csv(CSV_PATH)

    queries = []
    for _, row in new_data.iterrows():
        query = f"""
            INSERT INTO TrainData (age, weight, food_count, breed, gender, neutered, illness)
            VALUES ({row['age']}, {row['weight']}, {row['food_count']}, 
            '{row['breed']}', {row['gender']}, {row['neutered']}, {row['illness']});
        """
        queries.append(query)

    mysql_hook = MySqlHook(mysql_conn_id="meowmung_userdata")
    for query in queries:
        mysql_hook.run(query)


def train_model(**kwargs):
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        raise FileNotFoundError(f"Model not found")

    load_dotenv()

    mysql_key = os.getenv("MYSQL_KEY")
    mysql_conn_str = f"mysql+pymysql://root:{mysql_key}@localhost/meowmung_userdata"
    engine = create_engine(mysql_conn_str)

    query = """
        SELECT * FROM TrainData 
        WHERE train_id > (SELECT MAX(train_id) FROM TrainData WHERE model_trained = 1);
    """
    new_data = pd.read_sql(query, con=engine)

    if new_data.empty:
        return "No new data to train."

    X_new = new_data[["age", "weight", "food_count", "gender", "neutered", "breed"]]
    y_new = new_data["illness"]

    model.fit(X_new, y_new)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with engine.connect() as conn:
        conn.execute(
            """
            UPDATE TrainData SET model_trained = 1 WHERE train_id IN (%s)
        """,
            tuple(new_data["train_id"]),
        )


with DAG(
    dag_id="meowmung_userdata",
    schedule_interval="@monthly",
    default_args=default_args,
    tags=["meowmung", "user_data", "pet_health"],
    catchup=False,
) as dag:

    create_table = MySqlOperator(
        task_id="create_table",
        mysql_conn_id="meowmung_userdata",
        sql="""
            CREATE TABLE IF NOT EXISTS TrainData(
                train_id LONG PRIMARY KEY AUTO_INCREMENT,
                age INTEGER,
                weight FLOAT(3, 1),
                food_count FLOAT(3, 1),
                breed INTEGER,
                gender INTEGER,
                neutered BOOLEAN,
                illness INTEGER
            );
        """,
    )

    insert_data = PythonOperator(
        task_id="insert_data",
        python_callable=insert_csv,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    create_table >> insert_data >> train_model
