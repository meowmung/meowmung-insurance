from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") + ":5000"
MYSQL_HOST = os.getenv("MYSQL_HOST") + ":3306"
MODEL_STAGE = "Production"


def load_best_model(pet_type, MODEL_STAGE, metric_name, ascending=True):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        MODEL_NAME = f"best_clf_{pet_type}"
        client = MlflowClient()

        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        if not model_versions:
            raise ValueError(f"No registered models found for {MODEL_NAME}")

        production_models = [
            mv for mv in model_versions if mv.current_stage == MODEL_STAGE
        ]

        if not production_models:
            raise ValueError(f"No models in stage '{MODEL_STAGE}' for {MODEL_NAME}")

        model_metrics = []
        for model_version in production_models:
            run_id = model_version.run_id
            run_data = client.get_run(run_id).data
            metric_value = run_data.metrics.get(metric_name)
            if metric_value is not None:
                model_metrics.append((run_id, metric_value))

        if not model_metrics:
            raise ValueError(f"No metrics found for models in stage '{MODEL_STAGE}'")

        model_metrics.sort(key=lambda x: x[1], reverse=not ascending)
        best_run_id = model_metrics[0][0]

        model_uri = f"runs:/{best_run_id}/best_clf_{pet_type}"
        model = mlflow.sklearn.load_model(model_uri)

        print(f"Model (run_id: {best_run_id}) loaded successfully.")
        return model

    except Exception as e:
        print(f"Error loading best model: {str(e)}")
        raise


def pred_ill(pet_type, age, gender, breed, weight, food_count, neutered):

    model = load_best_model(
        pet_type, MODEL_STAGE, metric_name="accuracy", ascending=True
    )

    X = pd.DataFrame(
        [
            {
                "age": age,
                "weight": weight,
                "food_count": food_count,
                "breed_code": breed,
                "gender": gender,
                "neutered": neutered,
            }
        ]
    )
    predicted = model.predict(X)[0]

    predicted_code = int(predicted)

    return predicted_code


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
            host=MYSQL_HOST,
            port=3306,
            user="root",
            password="1234",
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
    disease: int


@app.post("/insurance/recommend", response_model=RecommendationResponse)
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

        return RecommendationResponse(disease=illness)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
