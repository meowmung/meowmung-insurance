from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bots.dataloader import *
from dotenv import load_dotenv
import os
import uvicorn
import boto3
from bots.summary import *

load_dotenv()

app = FastAPI()
s3 = boto3.client("s3")


class InfoRequest(BaseModel):
    terms: list
    file_path: str


@app.post("/insurance/admin")
async def save_summary(request: InfoRequest):
    try:
        terms = request.terms
        file_path = request.file_path

        if not terms or not isinstance(terms, list):
            raise HTTPException(status_code=400, detail="Invalid terms provided")

        if not file_path or not isinstance(file_path, str):
            raise HTTPException(status_code=400, detail="Invalid file_path provided")

        loader = Loader(file_path, terms)
        company = extract_company_name(file_path)

        save_summaries(loader, company)

        # bucket_name = os.getenv("S3_BUCKET")
        # s3_key = f"meowmung-insurance/data/dataloaders/{company}_loader.pkl"
        # s3.upload_file(loader, bucket_name, s3_key)

        # return {"message": "File uploaded successfully", "s3_key": s3_key}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("adminPage.adminApp:app", host="127.0.0.1", port=8008, reload=True)
