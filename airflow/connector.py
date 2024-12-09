from airflow.models import Connection
from airflow import settings
import os
from dotenv import load_dotenv


def create_connection():
    load_dotenv()
    conn = Connection(
        conn_id="meowmung_mysql",
        conn_type="mysql",
        host=os.getenv("MYSQL_HOST"),
        login="root",
        password="1234",
        schema="meowmung",
        port=3306,
    )
    session = settings.Session()
    session.add(conn)
    session.commit()
    print(f"Connection {conn.conn_id} created successfully!")


create_connection()
