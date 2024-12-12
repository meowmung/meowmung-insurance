import os
import boto3
import pickle

s3 = boto3.client("s3")


def load_loader(filepath):
    with open(filepath, "rb") as file:
        loader = pickle.load(file)

    return loader


loader = load_loader("data/dataloaders/DB_cat_loader_copy.pkl")

local_loader_path = f"/tmp/DB_cat_loader_copy.pkl"
loader.save_loader(local_loader_path)

bucket_name = os.getenv("S3_BUCKET")
s3_key = f"meowmung-insurance/data/dataloaders/DB_cat_loader_copy.pkl"
s3.upload_file(local_loader_path, bucket_name, s3_key)
