import pymysql
from bots.s3 import load_json_s3, load_yaml_s3


def mysql_connection():
    conn = pymysql.connect(
        host="localhost",
        user="lsj",
        password="1234",
        database="meowmung_test",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    return conn


def get_logo(company):
    insurance_logo = load_yaml_s3("logo")
    logo = insurance_logo.get(company)

    return logo


def generate_insurance_query(company):
    data = load_json_s3(company)
    logo = get_logo(company)
    insurance_id = data.get("company")
    company = insurance_id.split("_")[0]
    insurance = data.get("insurance")

    query_list = []

    table_query = """
    CREATE TABLE IF NOT EXISTS Insurance (
            insurance_id VARCHAR(25) PRIMARY KEY,
            company VARCHAR(25),
            insurance_item VARCHAR(50),
            logo VARCHAR(50)
        );
    """
    query_list.append(table_query)

    query_list.append(
        f"""INSERT INTO Insurance (insurance_id, company, insurance_item, logo)
            VALUES("{insurance_id}", "{company}", "{insurance}", "{logo}")
            ON DUPLICATE KEY UPDATE
                company = VALUES(company),
                insurance_item = VALUES(insurance_item),
                logo = VALUES(logo);"""
    )

    return query_list


def generate_term_query(company):
    data = load_json_s3(company)
    insurance_id = data.get("company")
    term_list = data.get("special_terms")

    query_list = []

    table_query = """
    CREATE TABLE IF NOT EXISTS Terms (
            term_id VARCHAR(25) PRIMARY KEY,
            insurance_id VARCHAR(50),
            term_name VARCHAR(50),
            term_causes VARCHAR(255),
            term_limits VARCHAR(255),
            term_details VARCHAR(255),
            INDEX idx_insurance_id (insurance_id),
            FOREIGN KEY (insurance_id) REFERENCES Insurance(insurance_id)
        );
    """
    query_list.append(table_query)

    for i, term in enumerate(term_list):
        term_id = f"{insurance_id}_term_{i}"
        term_name = term.get("name")
        summary = term.get("summary")
        term_causes = summary.get("causes")
        term_limits = summary.get("limits")
        term_details = summary.get("details")
        query_list.append(
            f"""INSERT INTO Terms (term_id, insurance_id, term_name, term_causes, term_limits, term_details)
                VALUES("{term_id}", "{insurance_id}", "{term_name}", "{term_causes}", "{term_limits}", "{term_details}")
                ON DUPLICATE KEY UPDATE
                    term_name = VALUES(term_name),
                    term_causes = VALUES(term_causes),
                    term_limits = VALUES(term_limits),
                    term_details = VALUES(term_details);"""
        )

    return query_list


def generate_results_query(company):
    data = load_json_s3(company)
    term_list = data.get("special_terms")
    insurance_id = data.get("company")

    query_list = []

    table_query = """
    CREATE TABLE IF NOT EXISTS Results (
            result_id BIGINT PRIMARY KEY AUTO_INCREMENT,
            term_id VARCHAR(25),
            disease_name VARCHAR(25),
            UNIQUE (term_id, disease_name),
            INDEX idx_term_id (term_id),
            FOREIGN KEY (term_id) REFERENCES Terms(term_id)
        );
    """
    query_list.append(table_query)

    for i, term in enumerate(term_list):
        term_id = f"{insurance_id}_term_{i}"
        illness_list = term.get("illness")
        for illness in illness_list:
            query_list.append(
                f"""INSERT INTO Results (term_id, disease_name)
                    VALUES("{term_id}", "{illness}")
                    ON DUPLICATE KEY UPDATE
                        disease_name = VALUES(disease_name);"""
            )

    return query_list


def execute_queries(queries):
    connection = mysql_connection()
    try:
        with connection.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
        connection.commit()
    except Exception as e:
        connection.rollback()
        print(f"Error executing queries: {e}")
    finally:
        connection.close()


def insert_insurances(company):
    queries = generate_insurance_query(company)
    execute_queries(queries)


def insert_terms(company):
    queries = generate_term_query(company)
    execute_queries(queries)


def insert_results(company):
    queries = generate_results_query(company)
    execute_queries(queries)


if __name__ == "__main__":
    company_list = [
        "DB_dog",
        "DB_cat",
        "KB_dog",
        "KB_cat",
        "meritz_dog",
        "meritz_cat",
        "samsung_dog",
        "samsung_cat",
        "hyundai_dog",
        "hyundai_cat",
    ]

    for company in company_list:
        insert_insurances(company)
        insert_terms(company)
        insert_results(company)
