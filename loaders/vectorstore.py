from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from loaders.dataloader import *
from chromadb.config import Settings
from chromadb import Client


class VectorStore(Chroma):
    def __init__(self, collection_name, client_settings, loader):
        super().__init__(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
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
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
        embeddings = embedding_function.embed_documents(doc_text)

        self.add_documents(documents=self.loader.docs, embeddings=embeddings)


def delete_collection(collection_name):
    client_settings = Settings(persist_directory="data/db", is_persistent=True)
    client = Client(client_settings)

    try:
        client.delete_collection(collection_name)
        print(f"컬렉션 '{collection_name}'이 성공적으로 삭제되었습니다.")
    except ValueError:
        print(f"컬렉션 '{collection_name}'이 존재하지 않습니다.")


def load_vectorstore(collection_name, loader):
    client_settings = Settings(persist_directory="data/db", is_persistent=True)

    client = Client(client_settings)
    collection = client.get_collection(collection_name)

    vectorstore = VectorStore(
        collection_name=collection_name,
        client_settings=client_settings,
        loader=loader,
    )

    vectorstore._collection = collection

    return vectorstore


if __name__ == "__main__":
    load_dotenv()

    pet_types = ["dog", "cat"]

    # for type in pet_types:
    #     loader = load_loader(f"data/dataloaders/{type}_loader.pkl")

    #     vectordb = load_vectorstore(f"{type}_store", loader)

    #     # vectordb.add_docs()

    #     print(vectordb._collection.count())

    for type in pet_types:
        loader = load_loader(f"data/dataloaders/{type}_loader.pkl")

        delete_collection(f"{type}_store")

        vectordb = VectorStore(
            collection_name=f"{type}_store",
            client_settings=Settings(persist_directory="data/db", is_persistent=True),
            loader=loader,
        )

        vectordb.add_docs()

        print(vectordb._collection.count())
