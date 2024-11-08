from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from loaders.dataloader import Loader
from chromadb.config import Settings


class VectorStore(Chroma):
    def __init__(self, collection_name, embedding_function, client_settings):
        super().__init__(
            collection_name=collection_name,
            embedding_function=embedding_function,
            client_settings=client_settings,
        )

    def add_docs(self, loader):
        """
        docs: list
            list of chunks of pdf to add (type: Documents)

        vectordb: chromadb.Collection
            db of embeddings

        returns: vectordb w/ new data added (skip existing data)
        """
        docs_tostr = [doc.page_content for doc in loader.docs]
        emb_func = OpenAIEmbeddings()
        embeddings = emb_func.embed_documents(docs_tostr)
        ids = loader.name_list

        existing_ids = set(self.get(ids=ids)["ids"])

        new_docs = []
        new_embeddings = []
        new_ids = []

        for i, doc_id in enumerate(ids):
            if doc_id not in existing_ids:
                new_docs.append(loader.docs[i])
                new_embeddings.append(embeddings[i])
                new_ids.append(doc_id)

        if new_docs:
            self.add_documents(
                documents=new_docs, embeddings=new_embeddings, ids=new_ids
            )
        else:
            print("no new docs - data with same IDs ignored")


if __name__ == "__main__":
    load_dotenv()

    vectordb = VectorStore(
        collection_name="pet-insurance",
        embedding_function=OpenAIEmbeddings(),
        client_settings=Settings(persist_directory="data/db", is_persistent=True),
    )

    print(vectordb._collection.count())

    loader = Loader("data/pdf")
    vectordb.add_docs(loader)

    print(vectordb._collection.count())
