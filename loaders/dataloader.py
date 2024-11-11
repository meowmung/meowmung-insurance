from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import pickle
from pathlib import Path


class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = doc_id

    def add_metadata(self, key, value):
        self.metadata[key] = value


class Loader:
    def __init__(self, dir_path=None, file_path=None):
        if file_path:
            self.docs = self.load_file(file_path)
        elif dir_path:
            self.docs = self.load_dir(dir_path)
        else:
            raise ValueError("Either 'dir_path' or 'file_path' must be provided.")

    def load_dir(self, dir_path, chunk_size=500, overlap=50):
        docs = []
        file_paths = glob(f"{dir_path}/*.pdf")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

        for path in file_paths:
            insurance_name = extract_insurance_name(path)
            loader = PyPDFLoader(path)
            doc_list = loader.load_and_split()
            for doc_index, doc in enumerate(doc_list):
                doc.metadata["insurance"] = insurance_name
                chunks = text_splitter.split_text(doc.page_content)
                for chunk_index, chunk in enumerate(chunks):
                    doc_id = f"{insurance_name}_{doc_index}_{chunk_index}"
                    docs.append(
                        Document(
                            page_content=chunk, metadata=doc.metadata, doc_id=doc_id
                        )
                    )

        return docs

    def load_file(self, file_path, chunk_size=500, overlap=50):
        docs = []
        insurance_name = extract_insurance_name(file_path)
        loader = PyPDFLoader(file_path)
        doc_list = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        for doc_index, doc in enumerate(doc_list):
            doc.metadata["insurance"] = insurance_name
            chunks = text_splitter.split_text(doc.page_content)
            for chunk_index, chunk in enumerate(chunks):
                doc_id = f"{insurance_name}_{doc_index}_{chunk_index}"
                docs.append(
                    Document(page_content=chunk, metadata=doc.metadata, doc_id=doc_id)
                )
        return docs

    def save_loader(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)


def load_loader(filepath):
    with open(filepath, "rb") as file:
        loader = pickle.load(file)

    return loader


def extract_insurance_name(file_path):
    insurance_name = Path(file_path).stem
    return insurance_name


if __name__ == "__main__":
    loader = load_loader("data/dataloaders/KB_dog_loader.pkl")
    for i in range(0, 30):
        print(i)
        print((loader.docs[i].page_content))
        print("=================================")
