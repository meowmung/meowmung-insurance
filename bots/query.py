import json
import glob
from airflow.providers.mysql.hooks.mysql import MySqlHook


def generate_insurance_query(file_path, table_name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    insurance_id = data.get("company")
    company = insurance_id.split("_")[0]
    insurance = data.get("insurance")

    query_list = []
    query_list.append(
        f"""INSERT INTO {table_name} (insurance_id, company, insurance_item)
            VALUES("{insurance_id}", "{company}", "{insurance}")
            ON DUPLICATE KEY UPDATE
                company = VALUES(company),
                insurance_item = VALUES(insurance_item);"""
    )

    return query_list


def generate_term_query(file_path, table_name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    insurance_id = data.get("company")
    term_list = data.get("special_terms")

    query_list = []
    for i, term in enumerate(term_list):
        term_id = f"{insurance_id}_term_{i}"
        term_name = term.get("name")
        summary = term.get("summary")
        term_causes = summary.get("causes")
        term_limits = summary.get("limits")
        term_details = summary.get("details")
        query_list.append(
            f"""INSERT INTO {table_name} (term_id, insurance_id, term_name, term_causes, term_limits, term_details)
                VALUES("{term_id}", "{insurance_id}", "{term_name}", "{term_causes}", "{term_limits}", "{term_details}")
                ON DUPLICATE KEY UPDATE
                    term_name = VALUES(term_name),
                    term_causes = VALUES(term_causes),
                    term_limits = VALUES(term_limits),
                    term_details = VALUES(term_details);"""
        )

    return query_list


def generate_results_query(file_path, table_name):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    term_list = data.get("special_terms")
    insurance_id = data.get("company")

    query_list = []
    for i, term in enumerate(term_list):
        term_id = f"{insurance_id}_term_{i}"
        illness_list = term.get("illness")
        for illness in illness_list:
            query_list.append(
                f"""INSERT INTO {table_name} (term_id, disease_name)
                    VALUES("{term_id}", "{illness}")
                    ON DUPLICATE KEY UPDATE
                        disease_name = VALUES(disease_name);"""
            )

    return query_list


def insert_insurances(dir_path, table_name, **kwargs):
    file_paths = glob.glob(dir_path)

    for path in file_paths:
        queries = generate_insurance_query(path, table_name)
        mysql_hook = MySqlHook(mysql_conn_id="meowmung_userdata")
        for query in queries:
            mysql_hook.run(query)


def insert_terms(dir_path, table_name, **kwargs):
    file_paths = glob.glob(dir_path)

    for path in file_paths:
        queries = generate_term_query(path, table_name)
        mysql_hook = MySqlHook(mysql_conn_id="meowmung_userdata")
        for query in queries:
            mysql_hook.run(query)


def insert_results(dir_path, table_name, **kwargs):
    file_paths = glob.glob(dir_path)

    for path in file_paths:
        queries = generate_results_query(path, table_name)
        mysql_hook = MySqlHook(mysql_conn_id="meowmung_userdata")
        for query in queries:
            mysql_hook.run(query)


if __name__ == "__main__":
    # ----------debug query--------
    query_list = generate_term_query("summaries/DB_cat_summary.json", "Insurance")

    for query in query_list:
        print(query)
