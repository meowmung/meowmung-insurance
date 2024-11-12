from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *


class RecommendBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def recommend(
        self,
        pet_type,
        breed,
        age,
        gender,
        neutered,
        concerned_illnesses,
    ):

        question = (
            "concerned_illnesses 배열에 나열된 질병들을 보장하는 특약들을 포함하는 "
            "보험 3개를 추천해주세요."
        )

        result = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in result])

        prompt = f"""
        당신은 반려견과 반려묘를 위한 보험 상품을 추천하는 챗봇입니다.
        사용자는 다음과 같은 특징을 입력했습니다:
        - 반려동물 종류: {pet_type}
        - 품종: {breed}
        - 나이: {age}
        - 성별: {gender}
        - 중성화 여부: {neutered}
        - 걱정되는 질병: {', '.join(concerned_illnesses)}

        위 정보를 바탕으로, 제공된 문서에서 이 동물에게 적합한 보험 상품을 3개 추천해주세요.
        각 보험에 대한 관련 특약만 포함하여 아래 JSON 형식으로 답변하세요:
        {{
            "insurance": "보험사 이름",
            "special_contracts": [context 에 주어진 json 형태의 각 특약의 세부정보]
        }}
        답변을 ```json 등의 래퍼로 감싸지 마세요.

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
        concerned_illnesses=["슬개골", "치과 치료"],
    )

    print(response.content)
