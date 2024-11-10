from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from loaders.vectorstore import *
from langchain_community.vectorstores import Chroma


class SummaryBot:
    def __init__(self, model_name, streaming, temperature, vectorstore):
        self.llm = ChatOpenAI(
            model_name=model_name, streaming=streaming, temperature=temperature
        )
        self.template = """ 
        당신은 보험 상품 설명서를 요약하는 챗봇입니다.
        사용자가 제공하는 설명서의 내용을 요약하여 다음의 정보를 제공합니다:
        1. 보험 회사 이름
        2. 보험 상품 명
        3. 세부 특약들 (최대한 자세히 기재하세요)
        
        제공된 문서 내용에만 기반하여 응답하세요. 임의의 정보를 생성하지 마세요.

        {context}

        질문: {question}
        답변:
        """
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})

    def summarize(self, question="보험 상품에 대한 요약을 제공해 주세요."):
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=self.template,
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type_kwargs={"prompt": prompt},
            retriever=self.retriever,
            return_source_documents=True,
        )

        response = qa_chain.invoke({"context": "", "query": question})
        print("Retrieved documents:", response.get("source_documents"))

        return response["result"]


if __name__ == "__main__":
    load_dotenv()

    vectordb = Chroma(
        collection_name="DB-store",
        persist_directory="data/db",
        embedding_function=OpenAIEmbeddings(),
    )

    bot = SummaryBot(
        model_name="gpt-4", streaming=False, temperature=0, vectorstore=vectordb
    )

    summary = bot.summarize()
    print(summary)
