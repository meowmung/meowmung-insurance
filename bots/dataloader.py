from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
from pathlib import Path
import re
import os
import fitz


class Document:
    def __init__(self, page_content, metadata=None, doc_id=None):
        self.page_content = page_content
        self.metadata = metadata or {}
        self.id = doc_id


class Loader:
    def __init__(self, file_path, terms):
        self.special_terms = self.extract_terms(file_path, terms)
        self.docs = self.load_file(file_path)

    def extract_terms(self, pdf_path, terms):
        print("extracting terms...")
        doc = fitz.open(pdf_path)
        extracted_terms = []

        for term in terms:
            page_num = term["page"] - 1
            term_name = normalize_text(term["term_name"])

            page = doc[page_num]

            target = page.search_for(term_name)[0]

            page.add_highlight_annot(target)
            extracted_terms.append(term_name)

        temp_path = pdf_path + ".temp"

        doc.save(temp_path, incremental=False)
        doc.close()

        os.replace(temp_path, pdf_path)

        return extracted_terms

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

                found_term = find_term(chunk, self.special_terms)
                if len(found_term) > 0:
                    newest_term = found_term[0]

                doc.metadata["term"] = newest_term
                docs.append(
                    Document(page_content=chunk, metadata=doc.metadata, doc_id=doc_id)
                )
        return docs


def normalize_text(text):
    text = text.replace("I", "Ⅰ")
    text = text.replace("\n", "")

    return text


def find_term(chunk, term_list):
    found_terms = []
    for term in term_list:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, chunk):
            found_terms.append(term)

    return found_terms


def extract_company_name(file_path):
    company_name = Path(file_path).stem
    return company_name
