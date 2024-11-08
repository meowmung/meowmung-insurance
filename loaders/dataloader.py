from langchain_community.document_loaders import PyPDFLoader
from glob import glob


class Loader:
    def __init__(self, dir_path):
        self.docs = self.load_pdf(dir_path)
        self.name_list = self.name_doc(self.docs)

    def load_pdf(self, dir_path):
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


if __name__ == "__main__":
    loader = Loader("data/pdf")
    # print(loader.docs[0])
    for i in range(len(loader.docs)):
        print(loader.docs[i].metadata["source"])
