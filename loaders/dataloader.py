from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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


class Loader:
    def __init__(self, dir_path=None, file_path=None, has_special_terms=True):
        if file_path:
            self.docs = self.load_file(file_path)
        elif dir_path:
            self.docs = self.load_dir_with_metadata(dir_path)
        else:
            raise ValueError("Either 'dir_path' or 'file_path' must be provided.")
        if has_special_terms:
            self.special_terms = extract_special_terms(self)

    def load_dir_with_metadata(self, dir_path):
        docs = []
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for chunk_idx, term in enumerate(data["special_terms"]):
                        doc = Document(
                            page_content=json.dumps(
                                term["details"], ensure_ascii=False
                            ),
                            metadata={
                                "company": data["company"],
                                "insurance": data["insurance"],
                                "term": term["name"],
                            },
                            doc_id=f"{filename}_chunk{chunk_idx}",
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

        newest_term = "None"

        for doc_index, doc in enumerate(doc_list):
            doc.metadata["company"] = company_name
            doc.metadata["term"] = "None"

            chunks = text_splitter.split_text(doc.page_content)
            for chunk_index, chunk in enumerate(chunks):
                doc_id = f"{company_name}_{doc_index}_{chunk_index}"

                found_term = find_term(chunk)
                if len(found_term) > 0:
                    newest_term = found_term[0]

                doc.metadata["term"] = newest_term
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


def find_term(chunk):
    pattern = r"특약 이름\s?\[([^\]]+)\]"
    match = re.findall(pattern, chunk)

    return match


if __name__ == "__main__":
    pet_types = ["dog", "cat"]

    for type in pet_types:
        loader = Loader(dir_path=f"summaries/{type}", has_special_terms=False)
        loader.save_loader(f"data/dataloaders/{type}_loader.pkl")

    # ____debug pdf load______
    # pet_type = "dog"
    # loader = load_loader(f"data/dataloaders/KB_dog_loader.pkl")

    # i = 26
    # print(loader.docs[i].page_content)
    # print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print(loader.docs[i].metadata)
    # print(f"++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
