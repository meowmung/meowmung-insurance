import json
import mysql.connector


def generate_illness_query(file_path, table_name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    insurance = data.get("insurance")
    term_list = data.get("special_terms")

    query_list = []
    for term in term_list:
        name = term.get("name")
        illness_list = term.get("illness")
        for illness in illness_list:
            query_list.append(
                f'INSERT INTO {table_name} VALUE("{insurance}", "{name}", "{illness}")'
            )

    return query_list


def generate_term_query(file_path, table_name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    term_list = data.get("special_terms")

    query_list = []
    for term in term_list:
        name = term.get("name")
        summary = term.get("summary")
        causes = summary.get("causes")
        limits = summary.get("limits")
        details = summary.get("details")
        query_list.append(
            f'INSERT INTO {table_name} VALUE("{name}", "{causes}", "{limits}", "{details}")'
        )

    return query_list


def insert(host, user, password, database, table_name, query_list):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database,
        )
        cursor = connection.cursor()

        for query in query_list:
            cursor.execute(query)

        connection.commit()
        print("Query executed successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed.")


if __name__ == "__main__":
    # ----------debug query--------
    pet_type = "dog"
    company = "KB_dog"
    file_path = f"summaries/{pet_type}/{company}_summary.json"

    query_list = generate_term_query(file_path, "TableName")

    for query in query_list:
        print(query)
