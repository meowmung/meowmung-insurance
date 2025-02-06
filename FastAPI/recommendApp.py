from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pymysql
import asyncio
from dotenv import load_dotenv
from bots.s3 import load_model_s3
from fastapi import FastAPI
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from prometheus_client import Counter
from starlette.responses import Response

load_dotenv()

app = FastAPI()
registry = CollectorRegistry()

MYSQL_HOST = os.getenv("MYSQL_HOST")


async def pred_ill(pet_type, age, gender, breed, weight, food_count, neutered):
    model = await asyncio.to_thread(load_model_s3, pet_type)

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

    predicted = await asyncio.to_thread(model.predict, X)
    predicted_code = int(predicted[0])
    return predicted_code


async def insert_info(
    pet_type, age, gender, breed, weight, food_count, neutered, current_disease
):
    load_dotenv()
    try:
        conn = await asyncio.to_thread(
            pymysql.connect,
            host=MYSQL_HOST,
            port=3306,
            user="lsj",
            password="1234",
            database="meowmung",
        )
        cursor = conn.cursor()

        query = f"""INSERT INTO TrainData_{pet_type} 
                    (age, weight, food_count, breed_code, gender, neutered, disease_code)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)"""

        await asyncio.to_thread(
            cursor.execute,
            query,
            (age, weight, food_count, breed, gender, neutered, current_disease),
        )
        conn.commit()

    except pymysql.MySQLError as e:
        raise HTTPException(status_code=500, detail=f"MySQL Error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    finally:
        if conn:
            conn.close()


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
        await insert_info(
            pet_type=request.pet_type,
            age=request.age,
            gender=request.gender,
            breed=request.breed,
            weight=request.weight,
            food_count=request.food_count,
            neutered=request.neutered,
            current_disease=request.current_disease,
        )

        illness = await pred_ill(
            pet_type=request.pet_type,
            age=request.age,
            gender=request.gender,
            breed=request.breed,
            weight=request.weight,
            food_count=request.food_count,
            neutered=request.neutered,
        )

        return RecommendationResponse(disease=illness)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


REQUEST_COUNT = Counter("api_requests_total", "Total API Requests", ["endpoint"])


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
