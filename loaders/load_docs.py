from loaders.dataloader import *
from loaders.vectorstore import *
from chromadb.config import Settings
from dotenv import load_dotenv
import glob


def load_by_insurance(file_path):
    insurance_name = extract_company_name(file_path)
    loader = Loader(file_path=file_path)
    loader.save_loader(f"data/dataloaders/{insurance_name}_loader.pkl")
    print(f"loader - {file_path}")


def store_by_insurance(file_path):
    insurance_name = extract_company_name(file_path)
    loader = load_loader(f"data/dataloaders/{insurance_name}_loader.pkl")
    delete_collection(f"{insurance_name}_store")
    vectordb = VectorStore(
        collection_name=f"{insurance_name}_store",
        client_settings=Settings(persist_directory="data/db", is_persistent=True),
        loader=loader,
    )
    vectordb.add_docs()
    print(f"store - {file_path}")


if __name__ == "__main__":

    load_dotenv()
    file_paths = glob.glob(f"data/pdf/*.pdf")

    for path in file_paths:
        load_by_insurance(path)
        store_by_insurance(path)
