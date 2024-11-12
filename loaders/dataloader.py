from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import pickle
from pathlib import Path
import re
import os
import json


class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = doc_id

    def add_metadata(self, key, value):
        self.metadata[key] = value


class Loader:
    def __init__(self, dir_path=None, file_path=None, has_special_terms=True):
        if file_path:
            self.docs = self.load_file(file_path)
        elif dir_path:
            self.docs = self.load_dir(dir_path)
        else:
            raise ValueError("Either 'dir_path' or 'file_path' must be provided.")
        if has_special_terms:
            self.special_terms = extract_special_terms(self)

    def load_dir(self, dir_path):
        docs = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc = Document(
                        page_content=json.dumps(data, ensure_ascii=False),
                        metadata={"source": filename},
                        doc_id=filename,
                    )
                    docs.append(doc)
        return docs

    def load_file(self, file_path, chunk_size=500, overlap=50):
        docs = []
        company_name = extract_company_name(file_path)
        loader = PyPDFLoader(file_path)
        doc_list = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        for doc_index, doc in enumerate(doc_list):
            doc.metadata["company"] = company_name
            chunks = text_splitter.split_text(doc.page_content)
            for chunk_index, chunk in enumerate(chunks):
                doc_id = f"{company_name}_{doc_index}_{chunk_index}"
                docs.append(
                    Document(page_content=chunk, metadata=doc.metadata, doc_id=doc_id)
                )
        return docs

    def save_loader(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)


def extract_special_terms(loader):
    special_terms = []
    pattern = r"특약 이름\s?\[([^\]]+)\]"

    for doc in loader.docs:
        matches = re.findall(pattern, doc.page_content)
        for match in matches:
            special_terms.append({"name": match})

    return special_terms


def load_loader(filepath):
    with open(filepath, "rb") as file:
        loader = pickle.load(file)

    return loader


def extract_company_name(file_path):
    company_name = Path(file_path).stem
    return company_name


if __name__ == "__main__":
    loader = Loader(dir_path="summaries/cat", has_special_terms=False)
    print(loader.docs[1].page_content)
    loader.save_loader("data/dataloaders/cat_loader.pkl")
