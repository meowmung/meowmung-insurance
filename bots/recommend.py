from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *
from collections import Counter


class RecommendBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever()

    def get_context(self, concerned_illnesses, top):
        retrieved_documents = []
        illness_synonyms = {
            "백내장": ["백내장", "녹내장", "안과", "시력"],
            "슬관절": ["슬관절", "슬개골", "컴퓨터"],
            "피부": ["피부", "약물"],
            "치과": ["치과", "구강", "컴퓨터"],
        }

        for illness in concerned_illnesses:
            question = f"{', '.join(illness_synonyms[illness])}을 포함하는 문서를 검색하세요. '{illness}' 라는 단어는 반드시 포함되어야 합니다."
            result = self.retriever.get_relevant_documents(question)

            filtered_result = []
            for i in range(len(result)):
                if illness in result[i].page_content:
                    filtered_result.append(result[i])

            retrieved_documents.extend(filtered_result)

        insurance_counts = Counter(
            [doc.metadata["insurance"] for doc in retrieved_documents]
        )
        top_insurances = [
            insurance for insurance, _ in insurance_counts.most_common(top)
        ]

        top_documents = [
            doc
            for doc in retrieved_documents
            if doc.metadata["insurance"] in top_insurances
        ]

        context = "\n".join(
            [
                f"보험사: {doc.metadata.get('company', '정보 없음')}\n"
                f"보험 상품: {doc.metadata.get('insurance', '정보 없음')}\n"
                f"특약: {doc.metadata.get('term', '정보 없음')}\n"
                f"{doc.page_content}"
                for doc in top_documents
            ]
        )

        return context

    def recommend(self, concerned_illnesses):
        context = self.get_context(concerned_illnesses, 2)
        prompt = f"""
        context 에 두 개의 보험 상품 정보를 아래의 형식으로 출력하세요.
        보험 상품과 특약의 이름은 metadata 를 참고하세요.
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
        model_name="gpt-4o-mini", streaming=False, temperature=0, vectorstore=vectordb
    )

    response = chatbot.recommend(concerned_illnesses=["백내장", "슬관절"])

    print(response.content)
