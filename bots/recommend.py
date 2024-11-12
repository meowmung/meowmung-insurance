from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *


class RecommendBot:

    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.template = PromptTemplate(
            input_variables=[
                "context",
                "question",
                "pet_type",
                "breed",
                "age",
                "gender",
                "neutered",
                "concerned_illnesses",
            ],
            template="""당신은 반려견과 반려묘를 위한 보험 상품을 추천하는 챗봇입니다.
            사용자는 다음과 같은 특징을 입력할 것입니다:
            1. 반려동물 종류: {pet_type}
            반려동물 종류에 맞는 보험 상품을 추천해야 합니다. 예를 들어, pet_type=dog 인데 DB_cat 처럼 고양이의 보험을 추천해선 안됩니다.
            2. 품종: {breed}
            3. 나이: {age}
            4. 성별: {gender}
            5. 중성화 여부: {neutered}
            6. 걱정되는 질병: {concerned_illnesses}

            입력된 특징을 사용하여 제공된 컨텍스트에서 가장 적합한 보험 상품과 특약을 찾으세요.
            제공되는 반려동물의 조건과 가장 유사한 특약을 가지는 서로 다른 회사의 보험 상품 3개를 추천해야 합니다.
            회사가 중복되어선 안됩니다.
            
            추천할 회사를 찾았다면, 해당 회사에서 제공하는 보험 상품의 특약들에 대한 정보를 제공하세요.
            특약들에 대한 정보는 제공된 context 에서 그대로 가져오면 됩니다.
            문서에서 제공되지 않는 단어는 출력되는 JSON 데이터 포함되어서는 안됩니다.

            다음 JSON 형식으로 응답하세요:
                "insurance": 출처 문서 파일 이름에 나와 있는 보험사 이름,
                "special_contracts": [입력된 특징과 가장 잘 맞는 특약 리스트]

            최종 답변에는 총 3개의 JSON 데이터가 포함됩니다.

            {context}

            질문: {question}
            답변:
            """,
        )

        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        self.qa_chain = LLMChain(prompt=self.template, llm=self.llm)

    def recommend(
        self,
        pet_type,
        breed,
        age,
        gender,
        neutered,
        concerned_illnesses,
        question="",
    ):
        result = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in result])
        response = self.qa_chain.invoke(
            {
                "pet_type": pet_type,
                "breed": breed,
                "age": age,
                "gender": gender,
                "neutered": neutered,
                "concerned_illnesses": concerned_illnesses,
                "context": context,
                "question": question,
            }
        )

        if isinstance(response, dict):
            response = {
                key: value
                for key, value in response.items()
                if key not in ["question", "text"]
            }

        return response


if __name__ == "__main__":
    load_dotenv()

    loader = load_loader("data/dataloaders/json_loader.pkl")
    vectordb = load_vectorstore(collection_name="json_store", loader=loader)

    chatbot = RecommendBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    response = chatbot.recommend(
        pet_type="dog",
        breed="닥스훈트",
        age="4",
        gender="F",
        neutered="yes",
        concerned_illnesses=["슬개골", "치과 치료"],
    )

    print(response)
