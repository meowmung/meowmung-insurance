from loaders.dataloader import *
from loaders.vectorstore import VectorStore
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings


def load_by_company(company_name):
    FILE_PATH = f"data/pdf/{company_name}.pdf"
    loader = Loader(file_path=FILE_PATH)
    loader.save_loader(f"data/dataloaders/{company_name}_loader.pkl")


def store_by_company(company_name):
    loader = load_loader(f"data/dataloaders/{company_name}_loader.pkl")
    vectordb = VectorStore(
        collection_name=f"{company_name}-store",
        embedding_function=OpenAIEmbeddings(),
        client_settings=Settings(persist_directory="data/db", is_persistent=True),
    )
    vectordb.add_docs(loader)
