from dotenv import load_dotenv
from bots.recommend import RecommendBot
from bots.summary import clean_json
from loaders.vectorstore import *
import pickle
import pandas as pd


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


def get_illness(concerned, age, gender, breed, weight, food_count, neutered):
    predicted = pred_ill(age, gender, breed, weight, food_count, neutered)
    if predicted not in concerned:
        concerned.append(predicted)

    return concerned


class AdvancedBot(RecommendBot):
    def __init__(self, model_name, streaming, temperature, vectorstore):
        super().__init__(model_name, streaming, temperature, vectorstore)

    def recommend(self, age, gender, breed, weight, food_count, neutered, concerned):
        illness = get_illness(
            concerned, age, gender, breed, weight, food_count, neutered
        )
        context = super().get_context(illness, 1)
        prompt = f"""
        context 에 나타난 보험 상품 정보와 추가 질문에 대한 답을 아래의 형식으로 출력하세요.
        보험 상품과 특약의 이름은 metadata 를 참고하세요.
        아래 형식으로 출력하세요:
        {{
            "insurance": "보험상품 이름",
            "special_contracts": [
            {{
                "name": 특약 이름,
                "causes": 보험금 지급 사유,
                "limits": 보장 한도,
                "details": 특약 요약
            }}
            ]
        }}
        최종 답변에는 insurance 는 무조건 1개여야 합니다. 각 insurance 의 special_contracts 배열의 길이는 제한이 없습니다.
        답변을 '''json 등의 래퍼로 감싸지 마세요
        {context}
        """

        response = self.llm(prompt)
        response_json = clean_json(response.content)
        print(type(response_json))

        return response_json


if __name__ == "__main__":
    load_dotenv()

    pet_type = "dog"
    loader = load_loader(f"data/dataloaders/{pet_type}_loader.pkl")
    vectordb = load_vectorstore(collection_name=f"{pet_type}_store", loader=loader)

    chatbot = AdvancedBot(
        model_name="gpt-4o-mini", streaming=False, temperature=0, vectorstore=vectordb
    )

    response = chatbot.recommend(
        age=5,
        gender=0,
        breed=1,
        weight=8,
        food_count=1,
        neutered=0,
        concerned=["슬관절"],
    )

    print(response)
