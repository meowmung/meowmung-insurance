from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from loaders.dataloader import *
from chromadb.config import Settings


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
    vectordb = VectorStore(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        client_settings=Settings(persist_directory="data/db", is_persistent=True),
        loader=loader,
    )

    return vectordb


if __name__ == "__main__":
    load_dotenv()

    loader = load_loader("data/dataloaders/KB_dog_loader.pkl")
    vectordb = load_vectorstore("KB_dog_store", loader)

    print(vectordb._collection.count())
