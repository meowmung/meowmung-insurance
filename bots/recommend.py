from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *


class RecommendBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})

    def recommend(
        self,
        pet_type,
        breed,
        age,
        gender,
        neutered,
        concerned_illnesses,
    ):

        illnesses_text = ", ".join(concerned_illnesses)

        question = f"{illnesses_text} 을 보장하는 특약들을 포함하는 서로 다른 보험 2개를 추천해주세요. 각 보험에 해당하는 특약들도 함께 추천해주세요."

        result = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in result])

        prompt = f"""
        사용자가 입력한 정보는 다음과 같습니다:
        - 동물 종류: {pet_type}, 품종: {breed}, 나이: {age}, 성별: {gender}, 중성화 여부: {neutered}
        - 걱정되는 질병: {illnesses_text}

        주어진 정보를 바탕으로 적합한 보험 상품 2개를 추천해주세요.
        보험 상품의 특약 중 details 에 걱정되는 질병에 대한 내용을 포함하는 것을 모두 검색하세요.
        특약의 갯수에는 제한이 없습니다. 단, 관련 없는 특약은 가져와서는 안됩니다.
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

    chatbot = RecommendBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    response = chatbot.recommend(
        pet_type=pet_type,
        breed="닥스훈트",
        age="4",
        gender="F",
        neutered="yes",
        concerned_illnesses=["백내장", "슬개골"],
    )

    print(response.content)
