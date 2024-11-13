from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from loaders.vectorstore import *
import json
import re


class SummaryBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.vectorstore = vectorstore
        self.retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(), llm=self.llm
        )

    def summarize(self, company):
        special_terms = self.vectorstore.loader.special_terms
        special_terms_name_list = [term["name"] for term in special_terms]
        insurance_info = {"company": company, "insurance": get_insurance(company)}
        special_terms_info = []

        for term_name in special_terms_name_list:
            question = f"{term_name}에 대한 정보를 context에서 찾아 반환하세요."
            print("retrieving")
            print("+++++++++++++++++++++++++++++++++++++++++++++++")
            term_result = self.retriever.get_relevant_documents(question)

            details = "\n".join(clean_text(doc.page_content) for doc in term_result)

            info = {
                "name": {term_name},
                "details": {details},
            }

            special_terms_info.append(info)
            print(f"{term_name} summarized")
            print("+++++++++++++++++++++++++++++++++++++++++++++++")

        insurance_info["special_terms"] = special_terms_info

        return insurance_info


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


def clean_json(text):
    cleaned_text = text.replace("\n", "").replace("    ", "").strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def clean_text(text):
    filtered_text = re.sub(r"[^가-힣0-9().]", " ", text)
    filtered_text = re.sub(r"\(\s*\)", "", filtered_text)
    cleaned_text = re.sub(r"\s+", " ", filtered_text).strip()

    return cleaned_text


def save_summaries(company):
    loader_path = f"data/dataloaders/{company}_loader.pkl"
    loader = load_loader(loader_path)
    vectordb = load_vectorstore(f"{company}_store", loader)

    bot = SummaryBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize(company)

    pet_type = company.split("_")[1]
    output_filename = f"summaries/{pet_type}/{company}_output.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(clean_json(summary["text"]), f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    load_dotenv()
    # file_paths = glob.glob(f"data/pdf/*.pdf")

    # for path in file_paths:
    #     company = extract_company(path)
    #     save_summaries(company)

    # ----- debug by file ------
    company = "DB_dog"
    loader_path = f"data/dataloaders/{company}_loader.pkl"
    loader = load_loader(loader_path)
    vectordb = load_vectorstore("DB_dog_store", loader)

    bot = SummaryBot(
        model_name="gpt-4o", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize(company)
    print(summary)

    # ----- debug saving -----
    # output_filename = f"data/json/summaries/{company}_output.json"
    # with open(output_filename, "w", encoding="utf-8") as f:
    #     json.dump(clean_json(summary["text"]), f, ensure_ascii=False, indent=4)
