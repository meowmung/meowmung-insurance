from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bots.dataloader import *
from dotenv import load_dotenv
from bots.summary import *
from bots.query import *
import uvicorn
from typing import List

load_dotenv()

app = FastAPI()


class Term(BaseModel):
    page: int
    term_name: str


class InfoRequest(BaseModel):
    terms: List[Term]
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

        insert_insurances(company)
        insert_terms(company)
        insert_results(company)

        return "Summary Pipeline Successful!"

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("adminPage.adminApp:app", host="0.0.0.0", port=8008, reload=True)
