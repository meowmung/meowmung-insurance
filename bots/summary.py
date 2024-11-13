from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from loaders.vectorstore import *
import json


class SummaryBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.vectorstore = vectorstore
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(), llm=self.llm
        )

    def summarize(self):
        question = (
            "특약들의 정보를 주어진 context 내에서 검색해주세요. "
            "검색된 모든 특약의 정보를 빠짐없이 기입해주세요."
        )
        result = self.retriever.get_relevant_documents(question)

        company = result[0].metadata["company"]
        insurance = get_insurance(company)
        special_terms = self.vectorstore.loader.special_terms

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

        prev_missing_terms = []
        max_attempts = 10
        attempt_count = 0

        while len(missing_terms) > 0 and attempt_count < max_attempts:
            print(f"누락된 특약: {missing_terms}")
            additional_docs = self.retriever.get_relevant_documents(
                f"누락된 특약: {', '.join(missing_terms)}에 대한 정보를 찾으세요."
            )

            additional_context = "\n".join(
                [doc.page_content for doc in additional_docs]
            )
            context += additional_context

            missing_terms = [
                term["name"] for term in special_terms if term["name"] not in context
            ]

            if missing_terms == prev_missing_terms:
                print("누락된 특약이 더 이상 변경되지 않으므로 반복을 종료합니다.")
                break

            prev_missing_terms = missing_terms
            attempt_count += 1

        prompt = f"""
        당신은 보험 상품 설명서를 요약하는 챗봇입니다.
        제공된 문서 내용을 요약하여 다음의 세 가지 정보를 JSON 형태로 제공합니다:
        1. 보험 회사 이름(company): {company}
        2. 보험 상품 명(insurance): {insurance}
        3. 세부 특약들(special terms): special_terms 배열에 있는 이름들은 문서에 반드시 존재하므로 모든 정보를 포함해야 합니다.
        특약의 이름이 유사해도 서로 다른 특약이니, 정보를 합치지 않고 따로 출력하세요.

        특약 정보는 다음과 같은 형식으로 제공하세요:
        {{
            "name": 특약 이름,
            "causes": 보험금 지급사유,
            "details": 보험금 지급 세부사항,
            "limit": 보상 금액 한도
        }}
        
        제공된 context 내에서 모든 특약 정보를 찾아 JSON 형태로 제공해주세요
        응답을 ```json 등의 래퍼로 감싸지 마세요:
        {context}
        """

        response = self.llm.invoke(prompt)
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
    return insurance_items.get(company)


def save_summaries(company):
    loader_path = f"data/dataloaders/{company}_loader.pkl"
    loader = load_loader(loader_path)
    vectorstore = load_vectorstore(f"{company}_store", loader)

    bot = SummaryBot(
        model_name="gpt-4o",
        streaming=False,
        temperature=0,
        vectorstore=vectorstore,
    )

    summary = bot.summarize()

    pet_type = company.split("_")[1]
    output_filename = f"summaries/{pet_type}/{company}_output.json"
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
    company_name = "meritz_dog"
    loader_path = f"data/dataloaders/{company_name}_loader.pkl"
    loader = load_loader(loader_path)
    vectordb = load_vectorstore("meritz_dog_store", loader)

    bot = SummaryBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize()
    # print(summary)
    print(clean_json(summary.content))

    # ----- debug saving -----
    # output_filename = f"data/json/summaries/{company_name}_output.json"
    # with open(output_filename, "w", encoding="utf-8") as f:
    #     json.dump(clean_json(summary["text"]), f, ensure_ascii=False, indent=4)
