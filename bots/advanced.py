from dotenv import load_dotenv
from bots.recommend import RecommendBot
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
    predicted_illness = model.predict(X)[0]
    # return int(predicted_illness)
    return "백내장"


class AdvancedBot(RecommendBot):
    def __init__(self, model_name, streaming, temperature, vectorstore):
        super().__init__(model_name, streaming, temperature, vectorstore)

    def recommend(
        self,
        concerned_illnesses,
        age,
        gender,
        breed,
        neutered,
        weight,
        food_count,
        additional_text,
    ):
        predicted_illness = pred_ill(age, gender, breed, weight, food_count, neutered)

        concerned_illnesses.append(predicted_illness)

        illnesses_text = ", ".join(concerned_illnesses)

        # ___debug illness___
        print("___debug illness___")
        print(illnesses_text)

        question = f"{illnesses_text} 을 보장하는 특약들을 포함한 최적의 보험 상품 1개를 추천해주세요. {illnesses_text} 에 대한 내용이 있는 특약들만 함께 추천해주세요. 특약의 상세 내용에 {illnesses_text} 이 포함되어야 합니다."

        if additional_text:
            question += (
                f" 추가로, {additional_text} 라는 질문에 대한 답변도 포함해 주세요."
            )
        else:
            additional_text = None

        # ___debug question___
        print("___debug question___")
        print(question)

        result = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in result])

        prompt = f"""
        사용자가 입력한 정보는 다음과 같습니다:
        - 걱정되는 질병: {illnesses_text}
        - 추가로 궁금한 사항: {additional_text}

        걱정되는 질병에 대한 내용을 포함하는 특약을 모두 찾고, 찾은 특약 중 가장 많이 등장한 보험상품 1개를 추천하세요.
        적합한 보험 상품은 위에 명시된 질병을 모두 보장할 수 있는 것입니다.
        각 보험 상품의 특약 정보는 special_contracts 배열에 추가하세요.
        사용자가 추가 질문을 제공한 경우, 해당 질문에 대한 답변을 additional_answer 키에 추가하세요.
        질문이 없으면 additional_answer 값은 "None"이어야 합니다.
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
            ],
            "additional_answer": "추가 질문에 대한 답변"
        }}
        최종 답변에는 insurance 는 무조건 1개여야 합니다. 각 insurance 의 special_contracts 배열의 길이는 제한이 없습니다.
        단, 명시된 질병에 대한 설명이 없는 특약은 절대로 가져와서는 안됩니다.
        답변 시 이전의 채팅 내역을 참고하지 마세요.
        답변을 '''json 등의 래퍼로 감싸지 마세요
        {context}
        """

        response = self.llm(prompt)
        return response


if __name__ == "__main__":
    load_dotenv()

    pet_type = "dog"
    loader = load_loader(f"data/dataloaders/{pet_type}_loader.pkl")
    vectordb = load_vectorstore(collection_name=f"{pet_type}_store", loader=loader)

    chatbot = AdvancedBot(
        model_name="gpt-4o-mini", streaming=False, temperature=0, vectorstore=vectordb
    )

    response = chatbot.recommend(
        concerned_illnesses=["치과"],
        age=5,
        gender=0,
        neutered=0,
        breed=1,
        weight=8,
        food_count=1,
        additional_text="'이 보험 상품을 추천한 이유도 알고 싶어요'",
    )

    print(response.content)

    # ______debug prediction_______
    # print(pred_ill(age=5, gender=0, neutered=0, breed=1, weight=8, food_count=1))
