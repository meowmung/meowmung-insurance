import json
import glob
from airflow.providers.mysql.hooks.mysql import MySqlHook
import pymysql

pymysql.install_as_MySQLdb()
from bots.s3 import load_json_s3, load_yaml_s3
from bots.dataloader import extract_company_name


def get_logo(company):
    insurance_logo = load_yaml_s3("logo")
    return insurance_logo.get(company)


def generate_insurance_query(company, table_name):
    data = load_json_s3(company)
    insurance_id = data.get("company")
    company = insurance_id.split("_")[0]
    insurance = data.get("insurance")
    logo = get_logo(company)

    query_list = []
    query_list.append(
        f"""INSERT INTO {table_name} (insurance_id, company, insurance_item)
            VALUES("{insurance_id}", "{company}", "{insurance}, {logo}")
            ON DUPLICATE KEY UPDATE
                company = VALUES(company),
                insurance_item = VALUES(insurance_item),
                logo = VALUES(logo);"""
    )

    return query_list


def generate_term_query(company, table_name):
    data = load_json_s3(company)
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


def generate_results_query(company, table_name):
    data = load_json_s3(company)
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
        company = extract_company_name(path)
        queries = generate_insurance_query(company, table_name)
        mysql_hook = MySqlHook(mysql_conn_id="meowmung_mysql")
        for query in queries:
            mysql_hook.run(query)


def insert_terms(dir_path, table_name, **kwargs):
    file_paths = glob.glob(dir_path)

    for path in file_paths:
        company = extract_company_name(path)
        queries = generate_term_query(company, table_name)
        mysql_hook = MySqlHook(mysql_conn_id="meowmung_mysql")
        for query in queries:
            mysql_hook.run(query)


def insert_results(dir_path, table_name, **kwargs):
    file_paths = glob.glob(dir_path)

    for path in file_paths:
        company = extract_company_name(path)
        queries = generate_results_query(company, table_name)
        mysql_hook = MySqlHook(mysql_conn_id="meowmung_mysql")
        for query in queries:
            mysql_hook.run(query)
