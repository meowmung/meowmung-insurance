from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
import mlflow


app = FastAPI()

MLFLOW_TRACKING_URI = "http://<mlflow-server-url>:5000"
MODEL_NAME = "ill_pred_rfclf"
MODEL_STAGE = "Production"


def load_model(pet_type):
    file_path = "models/ill_pred_rfclf.pkl"

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    return model

    # try:
    #     # 모델 로드 (해당 모델이 등록된 이름과 단계에 따라 호출)
    #     model = mlflow.pyfunc.load_model(
    #         model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    #     )
    #     print(f"Model {MODEL_NAME} loaded successfully.")
    #     return model
    # except Exception as e:
    #     print(f"Error loading model: {str(e)}")
    #     raise


def pred_ill(pet_type, age, gender, breed, weight, food_count, neutered):

    model = load_model(pet_type)

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
        pet_type = request.pet_type
        age = request.age
        gender = request.gender
        breed = request.breed
        weight = request.weight
        food_count = request.food_count
        neutered = request.neutered

        illness = pred_ill(
            pet_type=pet_type,
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
    uvicorn.run("FastAPI.advancedApp:app", host="127.0.0.1", port=8000, reload=True)
