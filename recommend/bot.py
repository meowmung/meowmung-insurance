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
        self.template = """You are a chatbot for recommending pet insurance products for dogs and cats.
        The user will input the following features:
        1. pet type : dog, cat
        2. breed
        3. age
        4. gender : M, F
        5. neuterized : yes, no
        6. concerned_illnesses : selected among [Patellar issues, Glaucoma, Dermatosis, Dental issues]

        Use the input features to find the most fitting insurance item from the provided context.
        The best insurance item should have special contracts (특약) that match the given features as closely as possible.

        Respond in the following JSON format:
            "insurance": The name of the insurance company as shown in the source document file name,
            "special_contracts": [List of special contracts that best match the features]
            "reason": reasons for your selctions

        {context}

        Question: {question}
        Answer:
        """

        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

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
            "Please recommend a pet insurance item for my dog. She is a Dachshund and is 4 years old. She is neutralized. I am concerned of Dental issues."
        )
    )
