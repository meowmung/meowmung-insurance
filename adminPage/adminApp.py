from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bots.dataloader import *
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class InfoRequest(BaseModel):
    terms: list
    file_path: str


@app.post("/insurance/admin")
async def save_loader(request: InfoRequest):
    try:
        terms = request.terms
        file_path = request.file_path

        loader = Loader(file_path, terms)

        company = extract_company_name(file_path)
        loader_path = f"data/dataloaders/{company}_loader.pkl"
        loader.save_loader(loader_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
