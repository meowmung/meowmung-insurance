from langchain_community.document_loaders import PyPDFLoader
from glob import glob
import pickle


class Loader:
    def __init__(self, dir_path=None, file_path=None):
        if file_path:
            self.docs = self.load_file(file_path)
        elif dir_path:
            self.docs = self.load_dir(dir_path)
        else:
            raise ValueError("Either 'dir_path' or 'file_path' must be provided.")
        self.name_list = self.name_doc(self.docs)

    def load_dir(self, dir_path):
        """
        dir_path: string
            path to dir containing pdf data

        returns: list of chunks of pdf (type: Documents)
        """
        docs = []
        file_paths = glob(f"{dir_path}/*.pdf")

        for path in file_paths:
            loader = PyPDFLoader(path)
            doc_list = loader.load_and_split()
            for doc in doc_list:
                docs.append(doc)

        return docs

    def load_file(self, file_path):
        docs = []

        loader = PyPDFLoader(file_path)
        doc_list = loader.load_and_split()
        for doc in doc_list:
            docs.append(doc)

    def name_doc(self, docs):
        """
        docs: list
            list of chunks of pdf (type: Documents)

        returns: list of names of chunks (will be used as doc ID)
        """
        name_list = []
        for i in range(len(docs)):
            source = docs[i].metadata["source"]
            page = docs[i].metadata["page"]
            name_list.append(f"{source}_page{page}")

        return name_list

    def save_loader(self, filepath):
        with open(filepath, "wb") as file:
            pickle.dump(self, file)


def load_loader(filepath):
    with open(filepath, "rb") as file:
        loader = pickle.load(file)

    return loader
