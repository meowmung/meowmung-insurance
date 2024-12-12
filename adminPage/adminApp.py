from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bots.dataloader import *
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


class InfoRequest(BaseModel):
    page: int
    term_name: str


@app.post("/insurance/admin")
async def save_loader(request: InfoRequest):
    try:
        page = request.page
        term_name = request.term_name

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
