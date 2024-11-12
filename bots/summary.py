from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *
from loaders.dataloader import extract_company_name
import json
import glob


class SummaryBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )

        self.template = PromptTemplate(
            input_variables=[
                "company",
                "insurance",
                "special_terms",
                "context",
                "query",
            ],
            template=""" 
            당신은 보험 상품 설명서를 요약하는 챗봇입니다.
            사용자가 제공하는 설명서의 내용을 요약하여 다음의 세 가지 정보를 JSON 형태로 제공합니다:
            1. 보험 회사 이름(company): {company}
            2. 보험 상품 명(insurance): {insurance}
            3. 세부 특수약관(특약)들(special terms): 보험 상품에 존재하는 모든 세부 특수약관(특약)들에 대한 정보를 검색하고, 각 약관에 대한 정보를 JSON 형태로 제공하세요.
            응답은 반드시 json 형태여야 합니다. ```json 등의 감싸기는 제거하세요.
            세부 특수 약관의 이름들의 목록은 다음과 같습니다: {special_terms}
            special_terms 배열에 있는 이름들은 반드시 문서에 존재하기 때문에, 반드시 찾아서 정보를 제공해야 합니다.
            괄호 안에 주어진 정보는 key 의 이름입니다.
            문서 내 명시된 모든 특약 이름과 정보들을 찾으세요.
            검색해야 할 정보는:
                * 특수약관의 이름 (name)
                * 보험금 지급사유 (causes)
                * 보험금 지급 세부사항 (details)
                * 보상 금액 한도 (limit)

            {context}

            질문: {query}
            답변:
            """,
        )
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever()
        self.qa_chain = LLMChain(prompt=self.template, llm=self.llm)

    def summarize(
        self,
        question="special_terms 에 명시된 모든 특약들의 정보를 주어진 context 내애서 정보를 검색해주세요. 누락되는 특약이 있으면 안됩니다.",
    ):
        result = self.retriever.get_relevant_documents(question)
        company = result[0].metadata["company"]
        insurance = get_insurance(company)

        special_terms = self.vectorstore.loader.special_terms
        print(special_terms)

        context = "\n".join(
            [
                doc.page_content
                for doc in result
                if any(term["name"] in doc.page_content for term in special_terms)
            ]
        )

        missing_terms = [
            term["name"] for term in special_terms if term["name"] not in context
        ]

        if missing_terms:
            print(f"누락된 특약: {missing_terms}")

        response = self.qa_chain.invoke(
            {
                "company": company,
                "insurance": insurance,
                "context": context,
                "special_terms": special_terms,
                "query": question,
            }
        )
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
        "meritz_cat": "(무)펫퍼민트 Cat&Family 보험 다이렉트",
        "samsung_dog": "무배당 삼성화재 다이렉트 반려견보험",
        "samsung_cat": "무배당 삼성화재 다이렉트 반려묘보험",
    }

    return insurance_items[company]


def save_summaries(company):
    loader_path = f"data/dataloaders/{company}_loader.pkl"
    loader = load_loader(loader_path)
    vectordb = load_vectorstore(f"{company}_store", loader)

    bot = SummaryBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize()

    output_filename = f"data/json/summaries/{company}_output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(clean_json(summary["text"]), f, ensure_ascii=False, indent=4)


def clean_json(text):
    cleaned_text = text.replace("\n", "").replace("    ", "").strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


if __name__ == "__main__":
    load_dotenv()
    # file_paths = glob.glob(f"data/pdf/*.pdf")

    # for path in file_paths:
    #     company = extract_company_name(path)
    #     save_summaries(company)

    # ----- debug by file ------
    company_name = "KB_dog"
    loader_path = f"data/dataloaders/{company_name}_loader.pkl"
    loader = load_loader(loader_path)
    vectordb = load_vectorstore("KB_dog_store", loader)

    bot = SummaryBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize()
    print(clean_json(summary["text"]))

    # ----- debug saving -----
    output_filename = f"data/json/summaries/{company_name}_output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(clean_json(summary["text"]), f, ensure_ascii=False, indent=4)
