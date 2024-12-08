from airflow.models import Connection
from airflow import settings


def create_connection():
    conn = Connection(
        conn_id="meowmung_mysql",
        conn_type="mysql",
        host="localhost",
        login="lsj",
        password="1234",
        schema="meowmung",
        port=3306,
    )
    session = settings.Session()
    session.add(conn)
    session.commit()
    print(f"Connection {conn.conn_id} created successfully!")


create_connection()
