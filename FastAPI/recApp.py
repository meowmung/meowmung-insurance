from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn


app = FastAPI()


def pred_ill(age, gender, breed, weight, food_count, neutered):
    file_path = "models/ill_pred_rfclf.pkl"

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    X = pd.DataFrame(
        [
            {
                "metadata_id_age": age,
                "metadata_physical_weight": weight,
                "metadata_breeding_food-amount": food_count,
                "encoded_metadata_id_breed": breed,
                "encoded_metadata_id_sex": gender,
                "neutered": neutered,
            }
        ]
    )
    predicted = model.predict(X)[0]
    return int(predicted)


class InfoRequest(BaseModel):
    pet_type: str
    age: int
    gender: int
    breed: int
    weight: int
    food_count: int
    neutered: int


class RecommendationResponse(BaseModel):
    illness: int


@app.post("/insurance/advanced", response_model=RecommendationResponse)
async def return_illness(request: InfoRequest):
    try:
        age = request.age
        gender = request.gender
        breed = request.breed
        weight = request.weight
        food_count = request.food_count
        neutered = request.neutered

        illness = pred_ill(
            age=age,
            gender=gender,
            breed=breed,
            weight=weight,
            food_count=food_count,
            neutered=neutered,
        )

        if isinstance(illness, int):
            return RecommendationResponse(illness=illness)
        else:
            raise ValueError("Invalid illness value returned from pred_ill")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("FastAPI.recApp:app", host="127.0.0.1", port=8000, reload=True)
