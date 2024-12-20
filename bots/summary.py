from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from bots.dataloader import *
import json
import re
from bots.s3 import load_yaml_s3, save_json_s3


class SummaryBot:
    def __init__(self, model_name, streaming, temperature, loader):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.loader = loader

    def summarize(self, company):
        special_terms = self.loader.special_terms
        print(special_terms)
        print("2")
        insurance_info = {"company": company, "insurance": get_insurance(company)}
        special_terms_info = []

        for term_name in special_terms:
            term_summary = self.retrieve_term_summary(term_name)
            illness = self.infer_illness(term_name, term_summary)
            info = {
                "name": clean_text(term_name),
                "summary": term_summary,
                "illness": illness,
            }
            special_terms_info.append(info)

        insurance_info["special_terms"] = special_terms_info
        return insurance_info

    def retrieve_term_summary(self, term_name):
        prompt = f"""
        아래 특약 이름에 대한 요약 정보를 작성하세요:
        특약 이름: {term_name}
        출력 형식:
        {{
            "details": {{
                "causes": "보험금 지급 사유",
                "limits": "보장 금액 한도",
                "details": "특약 요약"
            }}
        }}
        정보가 부족하다면, 지정된 형식과 특약 이름을 참고해 적절한 정보를 생성해 답하세요.
        """
        response = self.llm(prompt)

        try:
            details_json = json.loads(response.content)
            return details_json.get(
                "details",
                {
                    "causes": "보험금 지급 사유 없음",
                    "limits": "보장 금액 한도 없음",
                    "details": "특약 요약 없음",
                },
            )
        except json.JSONDecodeError:
            return {
                "causes": "기타 질병에 대해 폭 넓게 보장",
                "limits": "보험회사 홈페이지 참고",
                "details": "보험회사 홈페이지 참고",
            }

    def infer_illness(self, term_name, details):
        prompt = f"""
        아래 특약 이름과 설명에 따라 관련된 질병을 유추하세요:
        특약 이름: {term_name}
        특약 설명: {details}
        가능한 질병 카테고리: [백내장, 슬관절, 치과, 약물치료, 피부]
        특약 설명에 MRI 가 있다면, 관련된 질병은 슬관절과 치과입니다.
        출력 형식:
        {{
            "illness": 관련된 질병 카테고리 (하나 또는 여러 개 가능)를 list 로 반환
        }}
        illness 배열이 비어있다면, ["기타"] 로 응답하세요.
        """
        response = self.llm(prompt)
        try:
            illness_json = json.loads(response.content)
            return illness_json.get("illness", ["기타"])
        except json.JSONDecodeError:
            return ["기타"]


def get_insurance(company):
    insurance_items = load_yaml_s3("items")
    return insurance_items.get(company)


def clean_json(text):
    cleaned_text = text.replace("\n", "").replace("    ", "").strip()

    try:
        cleaned_text = f"[{cleaned_text.replace('}{', '},{')}]"
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def clean_text(text):
    filtered_text = re.sub(r"[^가-힣0-9().]", " ", text)
    filtered_text = re.sub(r"\(\s*\)", "", filtered_text)
    cleaned_text = re.sub(r"\s+", " ", filtered_text).strip()

    return cleaned_text


def save_summaries(loader, company):
    load_dotenv()

    bot = SummaryBot(
        model_name="gpt-4o-mini", streaming=False, temperature=0.3, loader=loader
    )

    summary = bot.summarize(company)

    save_json_s3(summary, company)
