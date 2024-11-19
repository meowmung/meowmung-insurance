from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bots.recommend import RecommendBot
from dotenv import load_dotenv
from loaders.dataloader import *
from loaders.vectorstore import *
import uvicorn


app = FastAPI()


class InfoRequest(BaseModel):
    pet_type: str
    concerned: list[str]


class RecommendationResponse(BaseModel):
    recommendation: list


load_dotenv()


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: InfoRequest):
    try:
        concerned = request.concerned
        pet_type = request.pet_type

        loader = load_loader(f"data/dataloaders/{pet_type}_loader.pkl")
        vectordb = load_vectorstore(collection_name=f"{pet_type}_store", loader=loader)

        bot = RecommendBot(
            model_name="gpt-4o-mini",
            streaming=False,
            temperature=0,
            vectorstore=vectordb,
        )
        response = bot.recommend(concerned_illnesses=concerned)

        return RecommendationResponse(recommendation=response)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("FastAPI.recommendApp:app", host="127.0.0.1", port=8000, reload=True)
