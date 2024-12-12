from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import pymysql
from dotenv import load_dotenv
import pickle

load_dotenv()

app = FastAPI()

MYSQL_HOST = os.getenv("MYSQL_HOST") + ":3306"


def load_model(pet_type):
    with open(f"data/models/best_clf_{pet_type}.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def pred_ill(pet_type, age, gender, breed, weight, food_count, neutered):

    model = load_model(pet_type)

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
