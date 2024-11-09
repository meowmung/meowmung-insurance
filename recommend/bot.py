from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *
from langchain_community.vectorstores import Chroma


class Chatbot:

    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.template = """당신은 반려견과 반려묘를 위한 보험 상품을 추천하는 챗봇입니다.
        사용자는 다음과 같은 특징을 입력할 것입니다:
        1. 반려동물 종류 (개, 고양이)
        2. 품종
        3. 나이
        4. 성별
        5. 중성화 여부
        6. 걱정되는 질병: [슬개골/슬관절, 녹내장/백내장, 피부염, 치과 치료] 중 선택

        입력된 특징을 사용하여 제공된 컨텍스트에서 가장 적합한 보험 상품과 특약을 찾으세요.
        제공되는 반려동물의 조건과 가장 유사한 특약을 가지는 보험 상품 3개를 추천해야 합니다.
        추천하는 특약의 갯수는 제한이 없습니다.
        특약 이름은 임의로 생성하지 않고, 주어진 문서에서 검색하여 사용자에게 있는 그대로 제공하세요.
        문서에서 제공되지 않는 단어는 출력되는 JSON 데이터 포함되어서는 안됩니다.

        다음 JSON 형식으로 응답하세요:
            "insurance": 출처 문서 파일 이름에 나와 있는 보험사 이름,
            "special_contracts": [입력된 특징과 가장 잘 맞는 특약 리스트]

        최종 답변에는 총 3개의 JSON 데이터가 포함됩니다.

        {context}

        질문: {question}
        답변:
        """

        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

    def ask(self, query):

        prompt = PromptTemplate(
            input_variables=["context", "question"], template=self.template
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type_kwargs={"prompt": prompt},
            retriever=self.retriever,
            return_source_documents=True,
        )

        response = qa_chain.invoke(query)

        # Debug
        print("Retrieved documents:", response.get("source_documents"))

        answer = response["result"]

        return answer


if __name__ == "__main__":
    load_dotenv()

    vectordb = Chroma(
        collection_name="pet-insurance",
        persist_directory="data/db",
        embedding_function=OpenAIEmbeddings(),
    )

    chatbot = Chatbot(
        model_name="gpt-4o-mini", streaming=False, temperature=0, vectorstore=vectordb
    )

    print(
        chatbot.ask(
            "제 강아지를 위한 보험 상품과 세부 특약들을 추천해주세요. 4살짜리 여자아이이며 견종은 닥스훈트입니다. 슬개골과 치과 치료가 걱정돼요."
        )
    )
