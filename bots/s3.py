import requests
from requests_aws4auth import AWS4Auth
import os
from dotenv import load_dotenv
import json
import pickle
from io import BytesIO
import yaml
import tempfile


load_dotenv()

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("AWS_DEFAULT_REGION")
service = os.getenv("service")
awsauth = AWS4Auth(aws_access_key, aws_secret_key, region, service)

bucket_url = "https://hk-project-1.s3.ap-northeast-2.amazonaws.com/meowmung-insurance/"


def load_model_s3(pet_type):
    url = bucket_url + f"data/models/best_clf_{pet_type}.pkl"
    response = requests.get(url, auth=awsauth)
    model = pickle.loads(response.content)

    return model


def load_json_s3(company):
    url = bucket_url + f"data/summaries/{company}_summary.json"
    response = requests.get(url, auth=awsauth)
    data = response.content.decode("utf-8")
    json_data = json.loads(data)

    return json_data


def load_pdf_s3(file_path):
    url = f"{bucket_url}{file_path}"
    response = requests.get(url, auth=awsauth)
    response.raise_for_status()

    pdf_file = BytesIO(response.content)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(pdf_file.getvalue())
        temp_pdf_path = temp_pdf.name

    print(f"File saved at {temp_pdf_path}")
    return temp_pdf_path


def load_yaml_s3(type):
    url = bucket_url + f"data/config/insurance_{type}.yaml"
    response = requests.get(url, auth=awsauth)

    yaml_data = yaml.safe_load(response.content)

    return yaml_data


def save_model_s3(model, pet_type):
    url = bucket_url + f"data/models/best_clf_{pet_type}.pkl"
    model_data = pickle.dumps(model)
    requests.put(url, data=model_data, auth=awsauth)


def save_json_s3(summary, company):
    output_filename = f"{company}_summary.json"
    url = bucket_url + f"data/summaries/{output_filename}"

    json_data = json.dumps(summary, ensure_ascii=False, indent=4)
    requests.put(
        url,
        data=json_data.encode("utf-8"),
        auth=awsauth,
        headers={"Content-Type": "application/json"},
    )


if __name__ == "__main__":
    print(load_pdf_s3("data/pdf/terms_test.pdf"))
