from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import uvicorn
import mlflow
import os
import pymysql
from dotenv import load_dotenv


app = FastAPI()

MLFLOW_TRACKING_URI = "http://<mlflow-server-url>:5000"
MODEL_STAGE = "Production"


def load_model(pet_type):
    file_path = "models/ill_pred_rfclf.pkl"

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    return model

    # try:
    #     # 모델 로드 (해당 모델이 등록된 이름과 단계에 따라 호출)
    #     MODEL_NAME = f"best_clf_{pet_type}"
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

    predicted_code = int(predicted)

    if pet_type == "dog":
        if predicted_code in [0, 1, 2]:
            return predicted_code
        elif predicted_code == 3:
            return 6
        elif predicted_code == 4:
            return 3

    if pet_type == "cat":
        if predicted_code == 0:
            return 0
        return predicted_code + 3


def insert_info(
    pet_type,
    age,
    gender,
    breed,
    weight,
    food_count,
    neutered,
    current_disease,
):
    load_dotenv()

    try:
        conn = pymysql.connect(
            host="localhost",
            port=3306,
            user="root",
            password=os.getenv("MYSQL_ROOT_PASSWORD"),
            database="meowmung",
        )
        cursor = conn.cursor()

        query = f"""INSERT INTO TrainData_{pet_type} 
                    (age, weight, food_count, breed_code, gender, neutered, disease_code)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)"""

        cursor.execute(
            query, (age, weight, food_count, breed, gender, neutered, current_disease)
        )
        conn.commit()
        print("Data inserted successfully!")

    except pymysql.MySQLError as e:
        print("MySQL error occurred:", e)

    except Exception as e:
        print("An error occurred:", e)

    finally:
        if conn:
            conn.close()
            print("Connection closed.")


class InfoRequest(BaseModel):
    pet_type: str
    age: int
    gender: int
    breed: int
    weight: float
    food_count: float
    neutered: int
    current_disease: int


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
        current_illness = request.current_disease

        insert_info(
            pet_type=pet_type,
            age=age,
            gender=gender,
            breed=breed,
            weight=weight,
            food_count=food_count,
            neutered=neutered,
            current_disease=current_illness,
        )

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
