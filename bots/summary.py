from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *


class SummaryBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )

        self.template = PromptTemplate(
            input_variables=["company", "insurance", "context", "query"],
            template=""" 
            당신은 보험 상품 설명서를 요약하는 챗봇입니다.
            사용자가 제공하는 설명서의 내용을 요약하여 다음의 세 가지 정보를 JSON 형태로 제공합니다:
            1. 보험 회사 이름: {company}
            2. 보험 상품 명: {insurance}
            3. 세부 특수약관(특약)들: 각 보험 상품마다 존재하는 세부 특수약관(특약)들에 대한 정보를 검색하고, 각 약관에 대한 정보를 JSON 형태로 제공하세요.
            약관이 여러 개면, 여러 개의 항목이 포함될 수 있습니다.
            괄호 안에 주어진 정보는 key 의 이름입니다.
            문서 내 명시된 모든 특약 이름과 정보들을 찾으세요.
            검색해야 할 정보는:
                * 특수약관의 이름 (name)
                * 보험금 지급사유 (causes)
                * 보험금 지급 세부사항 (details)
                * 보상 금액 한도 (limit)

            제공된 문서 내의 검색 결과에만 기반하여 응답하세요. 절대로 임의의 정보를 생성하지 마세요.
            특약 이름에 대한 응답 생성 시서로 다른 문장 내의 단어들을 조합하지 마세요.

            {context}

            질문: {query}
            답변:
            """,
        )
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 30})
        self.qa_chain = LLMChain(prompt=self.template, llm=self.llm)

    def summarize(
        self,
        question="주어진 context 내에 존재하는 모든 특약들에 대한 정보를 검색해주세요",
    ):
        result = self.retriever.get_relevant_documents(question)
        company = result[0].metadata["insurance"]
        insurance = get_insurance(company)

        special_terms = self.vectorstore.loader.special_terms

        context = "\n".join(
            [
                doc.page_content
                for doc in result
                if any(term["name"] in doc.page_content for term in special_terms)
            ]
        )

        response = self.qa_chain.invoke(
            {
                "company": company,
                "insurance": insurance,
                "context": context,
                "query": question,
            }
        )

        print(special_terms)
        print("+++++++++++++++++++++++++")
        return response


def get_insurance(company):
    insurance_items = {
        "DB_cat": "무배당 다이렉트 펫블리 반려묘보험",
        "DB_dog": "무배당 다이렉트 펫블리 반려견보험",
        "hyundai_dog": "무배당 현대해상다이렉트굿앤굿 우리펫보험",
        "hyundai_cat": "무배당 현대해상다이렉트굿앤굿 우리펫보험",
        "KB_dog": "KB 다이렉트 금쪽같은 펫보험 (강아지) (무배당)",
        "KB_cat": "KB 다이렉트 금쪽같은 펫보험 (고양이) (무배당)",
        "meritz_dog": "(무)펫퍼민트 Puppy&Family 보험 다이렉트",
        "mertiz_cat": "(무)펫퍼민트 Cat&Family 보험 다이렉트",
        "samsung_dog": "무배당 삼성화재 다이렉트 반려견보험",
        "samsung_cat": "무배당 삼성화재 다이렉트 반려묘보험",
    }

    return insurance_items[company]


if __name__ == "__main__":
    load_dotenv()

    loader = load_loader("data/dataloaders/KB_dog_loader.pkl")
    vectordb = load_vectorstore("KB_dog-store", loader)

    bot = SummaryBot(
        model_name="gpt-4-turbo", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize()
    print(summary["text"])
