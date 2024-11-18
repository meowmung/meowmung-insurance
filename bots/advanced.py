from dotenv import load_dotenv
from bots.recommend import RecommendBot
from loaders.vectorstore import *
import pickle
import numpy as np


def pred_ill(age, gender, neuterized, weight, food_count):
    file_path = "models/ill_pred_rfclf.pkl"

    with open(file_path, "rb") as f:
        model = pickle.load(f)

    predicted_illness = model.pred(
        age, gender, neuterized, weight, food_count
    ).to_list()[0]
    return predicted_illness


class AdvancedBot(RecommendBot):
    def __init__(self, model_name, streaming, temperature, vectorstore):
        super().__init__(model_name, streaming, temperature, vectorstore)

    def recommend(
        self,
        concerned_illnesses,
        age,
        gender,
        neuterized,
        weight,
        food_count,
        additional_text,
    ):
        predicted_illnesses = pred_ill(age, gender, neuterized, weight, food_count)

        all_illnesses = list(set(concerned_illnesses + predicted_illnesses))

        illnesses_text = ", ".join(all_illnesses)

        question = f"{illnesses_text} 을 보장하는 특약들을 포함하는 서로 다른 보험 2개를 추천해주세요. {illnesses_text} 에 대한 내용이 있는 특약들만 함께 추천해주세요. 특약의 상세 내용에 {illnesses_text} 이 포함되어야 합니다."

        if additional_text:
            question += f" 추가로, {additional_text}에 대한 답변도 포함해 주세요."

        result = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in result])

        prompt = f"""
        사용자가 입력한 정보는 다음과 같습니다:
        - 걱정되는 질병: {illnesses_text}
        - 추가로 궁금한 사항: {additional_text}

        걱정되는 질병에 대한 내용을 포함하는 특약을 모두 찾고, 찾은 특약 중 가장 많이 등장한 보험상품 2개를 추천하세요.
        적합한 보험 상품은 위에 명시된 질병을 모두 보장할 수 있는 것입니다.
        각 보험 상품의 특약 정보는 special_contracts 배열에 추가하세요.
        걱정되는 질병에 관한 내용을 포함하는 특약들을 모두 포함하여 아래 형식으로 출력하세요:
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
        최종 답변에는 insurance 는 무조건 2개여야 합니다. 각 insurance 의 special_contracts 배열의 길이는 제한이 없습니다.
        단, 명시된 질병에 대한 설명이 없는 특약은 절대로 가져와서는 안됩니다.
        답변 시 이전의 정보를 참고하지 마세요.
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
        concerned_illnesses=["피부", "치과"],
        age=5,
        gender="male",
        neuterized=True,
        weight=10,
        food_count=2,
        additional_text="이 보험 상품의 가입 조건을 알고 싶어요.",
    )

    print(response)
