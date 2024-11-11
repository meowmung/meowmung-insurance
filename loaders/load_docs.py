from loaders.dataloader import *
from loaders.vectorstore import VectorStore
from langchain_openai import OpenAIEmbeddings
from chromadb.config import Settings
from dotenv import load_dotenv


def load_by_insurance(file_path):
    insurance_name = extract_insurance_name(file_path)
    loader = Loader(file_path=file_path)
    loader.save_loader(f"data/dataloaders/{insurance_name}_loader.pkl")


def store_by_insurance(file_path):
    insurance_name = extract_insurance_name(file_path)
    loader = load_loader(f"data/dataloaders/{insurance_name}_loader.pkl")
    vectordb = VectorStore(
        collection_name=f"{insurance_name}_store",
        embedding_function=OpenAIEmbeddings(),
        client_settings=Settings(persist_directory="data/db", is_persistent=True),
        loader=loader,
    )
    vectordb.add_docs()


if __name__ == "__main__":
    load_dotenv()

    # file_paths = glob(f"data/pdf/*.pdf")
    # for path in file_paths:
    #     load_by_insurance(path)
    #     store_by_insurance(path)

    path = "data/pdf/KB_dog.pdf"
    load_by_insurance(path)
    store_by_insurance(path)
