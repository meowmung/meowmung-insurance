from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from loaders.dataloader import *
from chromadb.config import Settings
from chromadb import Client


class VectorStore(Chroma):
    def __init__(self, collection_name, embedding_function, client_settings, loader):
        super().__init__(
            collection_name=collection_name,
            embedding_function=embedding_function,
            client_settings=client_settings,
        )
        self.loader = loader

    def add_docs(self):
        """
        docs: list
            list of chunks of pdf to add (type: Documents)

        vectordb: chromadb.Collection
            db of embeddings

        returns: vectordb w/ new data added
        """
        doc_text = [doc.page_content for doc in self.loader.docs]
        embedding_function = OpenAIEmbeddings()
        embeddings = embedding_function.embed_documents(doc_text)

        self.add_documents(documents=self.loader.docs, embeddings=embeddings)


def load_vectorstore(collection_name, loader):
    """
    Collection을 로드하고 벡터 저장소를 반환합니다.
    이미 저장된 데이터를 불러오는 과정이므로 임베딩을 새로 계산하지 않습니다.
    """
    client_settings = Settings(persist_directory="data/db", is_persistent=True)

    client = Client(client_settings)
    collection = client.get_collection(collection_name)

    vectorstore = VectorStore(
        collection_name=collection_name,
        embedding_function=None,
        client_settings=client_settings,
        loader=loader,
    )

    vectorstore._collection = collection

    return vectorstore


if __name__ == "__main__":
    load_dotenv()

    pet_types = ["dog", "cat"]

    for type in pet_types:
        loader = load_loader(f"data/dataloaders/{type}_loader.pkl")

        vectordb = load_vectorstore(f"{type}_store", loader)

        # vectordb.add_docs()

        print(vectordb._collection.count())
