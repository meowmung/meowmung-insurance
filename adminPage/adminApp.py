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

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
