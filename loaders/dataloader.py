from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from glob import glob
import pickle


class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


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
            loader = PyPDFLoader(path)
            doc_list = loader.load_and_split()
            for doc in doc_list:
                chunks = text_splitter.split_text(doc.page_content)
                for chunk in chunks:
                    docs.append(Document(page_content=chunk, metadata=doc.metadata))

        return docs

    def load_file(self, file_path, chunk_size=500, overlap=50):
        docs = []
        loader = PyPDFLoader(file_path)
        doc_list = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        for doc in doc_list:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                docs.append(Document(page_content=chunk, metadata=doc.metadata))

        return docs

    def save_loader(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)


def load_loader(filepath):
    with open(filepath, "rb") as file:
        loader = pickle.load(file)

    return loader


if __name__ == "__main__":
    loader_list = glob("data/dataloaders/*.pkl")

    for loader_path in loader_list:
        loader = load_loader(loader_path)
        print(loader_path)
        print((loader.docs[0].page_content))
        print("===============\n")
